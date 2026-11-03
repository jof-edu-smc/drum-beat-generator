import os
import hmac
import time
from collections import defaultdict, deque
from pathlib import Path
from dotenv import load_dotenv
from models.gan import Generator, Conditioner, Discriminator
from utilities.generate import generate_music
from utilities.files import save_gan_tensor_to_drums
from flask import Flask, abort, jsonify, request, send_from_directory # type: ignore
from flask_cors import CORS # type: ignore

load_dotenv()
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = str(APP_DIR)
API_KEY = os.getenv("API_KEY")
RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "60"))
RATE_WINDOW_SECONDS = 60
request_times: defaultdict[str, deque[float]] = defaultdict(deque)

# FIX: Explicitly pass absolute paths to Flask so it never misses the target inside Docker
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
CORS(app, resources={r"/api/*": {"origins": [origin.strip() for origin in allowed_origins]}})


@app.before_request
def protect_api():
    if not request.path.startswith("/api/") or request.path == "/api/health":
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
def generate():
    # ===========================================
    # genre: genre.toLowerCase(),
    # creativity: parseFloat(creativity),
    # midi_filename: cleanedFilename,
    # num_bars: parseInt(numBars, 10),
    # ===========================================
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    
    try:
        genre = payload.get("genre", "all")
        creativity = float(payload.get("creativity", 1.0))
        midi_filename = payload.get("midi_filename", "generated_sample")
        num_bars = int(payload.get("num_bars", 4))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid generation parameters."}), 400
    
    return jsonify({"message": "This endpoint is not yet implemented."}), 501
    
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path: str):
    print(f"Received request for path: {path} at the time of {os.path.abspath(__file__)}")
    return send_from_directory(STATIC_DIR, 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=True)