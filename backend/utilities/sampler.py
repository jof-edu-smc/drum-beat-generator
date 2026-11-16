# Exports a list of audio files from GCS to be used in the Tone.js sampler for use in app.py. 
# The function `get_audio_files_for_sampler` takes a list of file names and generates signed URLs for each audio file stored in the specified GCS bucket. 
# These signed URLs allow the frontend to access the audio files securely without exposing the actual GCS bucket URL.

import os
import io
from pathlib import Path
from flask import send_file, jsonify # type: ignore
from google.cloud import storage # type: ignore

def get_audio_file_for_sampler(file_name, drum_bucket_name):
    """
    Given a list of file names, generate signed URLs for each audio file stored in the GCS bucket.
    This function is used to provide the frontend with access to the audio files for the Tone.js sampler.

    Args:
        file_name (list): The name of the audio file to retrieve from GCS.
        drum_bucket_name (str): The name of the GCS bucket containing the audio files.
        storage_client: The GCS storage client.

    Returns:
        dict: A dictionary containing the signed URLs for each audio file.
    """ 
    storage_client = storage.Client() 
    # 1. Connect to GCS and reference the file
    bucket = storage_client.bucket(drum_bucket_name)
    blob = bucket.blob(file_name)
    # 2. Check if the file actually exists in GCS
    if not blob.exists():
        return jsonify({"error": f"File {file_name} not found in GCS."}), 404

    # 3. Download the file contents into an in-memory byte buffer
    midi_data = io.BytesIO()
    blob.download_to_file(midi_data)
    midi_data.seek(0)  # Reset pointer to the start of the bytes buffer

    # 4. Stream the raw file bytes back to the frontend with the correct MIDI header
    return send_file(
        midi_data,
        mimetype="audio/midi",
        as_attachment=False,  # Set to True if you want to force browser download
        download_name=file_name
    )