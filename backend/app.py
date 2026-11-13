import os
import hmac
import time
import torch # type: ignore
import pprint 
import tempfile
import urllib.request
import urllib.error
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from dotenv import load_dotenv # type: ignore
from google.auth.transport import requests as google_requests # type: ignore
from google.oauth2 import id_token as google_id_token # type: ignore
from models.gan import Generator, Conditioner
from utilities.generate import generate_music
from utilities.files import (
    append_gan_tensor_to_drums,
    save_gan_tensor_to_drums,
    generate_gcs_url,
    get_weights_from_gcs,
    process_midi_to_tensor_list,
)
from flask import Flask, Response, abort, jsonify, request, send_from_directory, url_for # type: ignore
from flask_cors import CORS # type: ignore

pp = pprint.PrettyPrinter(indent=4)

load_dotenv()
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = str(APP_DIR)
API_KEY = os.getenv("DRUMBEAT_API_KEY") or os.getenv("API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GENERATION_SERVICE_URL = os.getenv(
    "GENERATION_SERVICE_URL",
    "http://127.0.0.1:5001/api/internal/generate",
)
RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "60"))
RATE_WINDOW_SECONDS = 60

LATENT_DIM = int(os.getenv("LATENT_DIM", "100"))
MIN_PITCH = int(os.getenv("MIN_PITCH", "22"))
MAX_PITCH = int(os.getenv("MAX_PITCH", "59"))
W = int(os.getenv("W", "16"))
CHORD_DIM = int(os.getenv("CHORD_DIM", "0"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR='checkpoints/' # DO NOT MODIFY Directory to save model checkpoints.
PRETAINED_DIR = os.getenv("PRETRAINED_DIR", 'weights/')
DEV = os.getenv("DEV", "False").lower() in {"1", "true", "yes", "on"}

request_times: defaultdict[str, deque[float]] = defaultdict(deque)

# FIX: Explicitly pass absolute paths to Flask so it never misses the target inside Docker
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
CORS(app, resources={r"/api/*": {"origins": [origin.strip() for origin in allowed_origins]}})


def authenticate_user():
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return jsonify({"error": "Google authentication is required."}), 401
    if not GOOGLE_CLIENT_ID:
        app.logger.error("GOOGLE_CLIENT_ID is not configured; refusing user authentication")
        return jsonify({"error": "User authentication is not configured."}), 503

    token = authorization.removeprefix("Bearer ").strip()
    try:
        google_id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError:
        app.logger.warning("Invalid Google authentication token")
        return jsonify({"error": "Invalid Google authentication token."}), 401
    return None


@app.before_request
def protect_api():
    if (
        request.method == "OPTIONS"
        or not request.path.startswith("/api/")
        or request.path == "/api/generate"
        or request.path == "/api/health"
        or (DEV and request.path.startswith("/api/midi/"))
    ):
        return None

    now = time.monotonic()
    client = request.remote_addr or "unknown"
    recent_requests = request_times[client]
    while recent_requests and now - recent_requests[0] >= RATE_WINDOW_SECONDS:
        recent_requests.popleft()
    if len(recent_requests) >= RATE_LIMIT:
        return jsonify({"error": "Too many requests. Try again later."}), 429
    recent_requests.append(now)

    if not API_KEY:
        app.logger.error("API_KEY is not configured; refusing protected API access")
        return jsonify({"error": "API is not configured for authenticated access."}), 503

    supplied_key = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(supplied_key, API_KEY):
        app.logger.warning("Unauthorized API access attempt from %s", client)
        return jsonify({"error": "Unauthorized"}), 401
    return None


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health():
    return jsonify({"message": "API is online."}), 200

@app.route("/api/generate", methods=["POST"])
def frontend_generate():
    """Forward browser requests without exposing the service API key."""
    authentication_error = authenticate_user()
    if authentication_error:
        return authentication_error
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    if not API_KEY:
        app.logger.error("API_KEY is not configured; refusing BFF request")
        return jsonify({"error": "API is not configured."}), 503

    upstream_request = urllib.request.Request(
        GENERATION_SERVICE_URL,
        data=jsonify(payload).get_data(),
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(upstream_request, timeout=60) as upstream_response:
            return Response(
                upstream_response.read(),
                status=upstream_response.status,
                content_type=upstream_response.headers.get_content_type(),
            )
    except urllib.error.HTTPError as error:
        return Response(
            error.read(),
            status=error.code,
            content_type=error.headers.get_content_type(),
        )
    except urllib.error.URLError:
        app.logger.exception("Generation service is unavailable")
        return jsonify({"error": "Generation service is unavailable."}), 502

@app.route("/api/internal/generate", methods=["POST"])
def generate():
    # ===========================================
    # genre: genre.toLowerCase(),
    # creativity: parseFloat(creativity),
    # midi_filename: cleanedFilename,
    # num_bars: parseInt(numBars, 10),
    # midi_url: midiUrl || ''
    # ===========================================
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    
    try:
        genre = payload.get("genre", "all").lower()
        creativity = float(payload.get("creativity", 1.0))
        midi_filename = Path(
            str(payload.get("midi_filename", "generated_sample"))
        ).name
        num_bars = int(payload.get("num_bars", 4))
        midi_url = payload.get("midi_url", None)
        
        generator = Generator(
            min_pitch=MIN_PITCH,
            max_pitch=MAX_PITCH,
            w=W,
            latent_dim=LATENT_DIM,
            chord_dim=CHORD_DIM,
        )
        conditioner = Conditioner(
            min_pitch=MIN_PITCH, 
            max_pitch=MAX_PITCH,
            w=W
        )
        generator.to(DEVICE)
        conditioner.to(DEVICE)
        
        source_midi_path = None
        if midi_url:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_file:
                source_midi_path = temp_file.name
            try:
                urllib.request.urlretrieve(midi_url, source_midi_path)
                custom_conditions = process_midi_to_tensor_list(
                    midi_path=source_midi_path,
                    target_bpm=120,
                    time_signature=(4, 4),
                    stride=W,
                    device=DEVICE,
                )
                if not isinstance(custom_conditions, list) or not custom_conditions:
                    raise ValueError("The conditioning MIDI does not contain a complete bar.")
                print("Applying the final bar as a custom condition from MIDI file:", midi_url)
            except (urllib.error.URLError, OSError, ValueError):
                if source_midi_path and os.path.exists(source_midi_path):
                    os.unlink(source_midi_path)
                raise
        else:
            custom_conditions = None
        
        generated_bars = generate_music(
            generator=generator,
            conditioner=conditioner,
            opt=type("Opt", (), {
                "min_pitch": MIN_PITCH,
                "max_pitch": MAX_PITCH,
                "w": W,
                "latent_dim": LATENT_DIM,
                "chord_dim": CHORD_DIM,
                "genre": genre
            })(),
            device=DEVICE,
            NUM_BARS_TO_GENERATE=num_bars,
            CHORD_PROGRESSION=None,
            checkpoint_path=os.path.join(PRETAINED_DIR, f"{genre}_model_weights.pth"),
            creativity=creativity,
            custom_conditions=custom_conditions
        )
        
        output_path = (
            str(APP_DIR / f"{midi_filename}.mid")
            if DEV
            else f"{midi_filename}.mid"
        )
        if source_midi_path:
            file_location = append_gan_tensor_to_drums(
                base_midi_path=source_midi_path,
                tensor_input=generated_bars,
                output_path=output_path,
                bpm=120,
                is_dev=DEV,
            )
        else:
            file_location = save_gan_tensor_to_drums(
                tensor_input=generated_bars,
                output_path=output_path,
                bpm=120,
                is_dev=DEV,
            )
        if source_midi_path and os.path.exists(source_midi_path):
            os.unlink(source_midi_path)
        if DEV:
            return jsonify({
                "message": f"Generated a bar of music at {len(generated_bars)} in the location: {file_location}",
                "signed_url": url_for(
                    "serve_generated_midi",
                    filename=Path(file_location).name,
                    _external=True,
                )
            }), 200
        else:
            return jsonify({
                "message": f"Generated a bar of music at {len(generated_bars)} in the location: {file_location}",
                "signed_url": generate_gcs_url(file_location)
            }), 200
    
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid generation parameters."}), 400


@app.get("/api/midi/<path:filename>")
def serve_generated_midi(filename: str):
    if not DEV:
        abort(404)
    return send_from_directory(APP_DIR, filename, mimetype="audio/midi")
    
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path: str):
    print(f"Received request for path: {path} at the time of {os.path.abspath(__file__)}")
    return send_from_directory(STATIC_DIR, 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=True)