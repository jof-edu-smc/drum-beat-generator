import io
from flask import send_file, jsonify # type: ignore
from google.cloud import storage # type: ignore

def get_midi_file_from_gcs(filename, bucket_name):   
    storage_client = storage.Client()
    # 1. Connect to GCS and reference the file
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    
    # 2. Check if the file actually exists in GCS
    if not blob.exists():
        return jsonify({"error": f"File {filename} not found in GCS."}), 404

    # 3. Download the file contents into an in-memory byte buffer
    midi_data = io.BytesIO()
    blob.download_to_file(midi_data)
    midi_data.seek(0)  # Reset pointer to the start of the bytes buffer

    # 4. Stream the raw file bytes back to the frontend with the correct MIDI header
    return send_file(
        midi_data,
        mimetype="audio/midi",
        as_attachment=False,  # Set to True if you want to force browser download
        download_name=filename
    )