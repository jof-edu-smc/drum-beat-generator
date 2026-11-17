# Exports a list of audio files from GCS to be used in the Tone.js sampler for use in app.py. 
# The function `get_audio_files_for_sampler` takes a list of file names and generates signed URLs for each audio file stored in the specified GCS bucket. 
# These signed URLs allow the frontend to access the audio files securely without exposing the actual GCS bucket URL.

import mimetypes
import io
from pathlib import Path
from flask import send_file, jsonify # type: ignore
from google.cloud import storage # type: ignore

def get_audio_file_for_sampler(file_name, drum_bucket_name):
    """
    Retrieves and streams audio sample files from GCS for the Tone.js sampler
    with correct MIME types and caching headers.
    """
    storage_client = storage.Client() 
    bucket = storage_client.bucket(drum_bucket_name)
    blob = bucket.blob(file_name)

    if not blob.exists():
        return jsonify({"error": f"File {file_name} not found in GCS."}), 404

    # Download file contents into an in-memory byte buffer
    audio_data = io.BytesIO()
    blob.download_to_file(audio_data)
    audio_data.seek(0)

    # Dynamically determine the correct audio MIME type (e.g., audio/flac, audio/wav)
    mimetype, _ = mimetypes.guess_type(file_name)
    if not mimetype or not mimetype.startswith('audio/'):
        mimetype = 'audio/flac' # Safe fallback for your sampler samples

    # Stream file bytes back with proper mimetype and caching to prevent 429s
    response = send_file(
        audio_data,
        mimetype=mimetype,
        as_attachment=False,
        download_name=Path(file_name).name
    )
    
    # Cache samples in the browser for 24 hours to prevent hammering the proxy
    response.headers["Cache-Control"] = "public, max-age=86400, immutable"
    return response