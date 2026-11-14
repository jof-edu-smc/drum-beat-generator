"""Flask API service for the drumbeat GAN training pipeline using minimal environment configuration and API key auth."""

from __future__ import annotations

import json
import time
import itertools
import logging
import os
import random
import tempfile
from pathlib import Path

import numpy as np # type: ignore
import torch # type: ignore
from flask import Flask, jsonify, request # type: ignore
from google.cloud import storage # type: ignore
from torch.utils.data import DataLoader # type: ignore

from datasets.dataset import DrumDataset
from models.gan import Conditioner, Discriminator, Generator
from utilities.train import train

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)

# Only the 4 requested environment variables are read here
TRAINING_DATA_BUCKET = os.getenv("TRAINING_DATA_BUCKET", "drumbeat-studio-training-data")
MODEL_WEIGHTS_BUCKET = os.getenv("MODEL_WEIGHTS_BUCKET", "drumbeat-studio-model-training-data")
MIDI_PREFIX = os.getenv("MIDI_PREFIX", "Apple").strip("/")

LOCAL_CHECKPOINT_DIR = "/tmp/checkpoints"
os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)

def upload_weight_to_gcs(local_path, destination_blob_name):
    """Uploads a trained model checkpoint to the model weights GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(MODEL_WEIGHTS_BUCKET)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    LOGGER.info("Successfully uploaded %s to gs://%s/%s", local_path, MODEL_WEIGHTS_BUCKET, destination_blob_name)


def download_weight_from_gcs(source_blob_name, local_path):
    """Downloads an existing model checkpoint from the model weights GCS bucket for fine-tuning."""
    client = storage.Client()
    bucket = client.bucket(MODEL_WEIGHTS_BUCKET)
    blob = bucket.blob(source_blob_name)
    if blob.exists():
        blob.download_to_filename(local_path)
        LOGGER.info("Successfully downloaded gs://%s/%s to %s", MODEL_WEIGHTS_BUCKET, source_blob_name, local_path)
        return True
    LOGGER.info("No existing checkpoint found at gs://%s/%s. Starting fresh.", MODEL_WEIGHTS_BUCKET, source_blob_name)
    return False


def run_training(
    metadata_path: str,
    checkpoint_dir: str,
    genre: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    latent_dim: int,
    min_pitch: int,
    max_pitch: int,
    w: int,
    lambda_1: float,
    lambda_2: float,
    pin_memory: bool,
    num_workers: int,
    fine_tuned: bool = False,
) -> None:
    """Train the GAN inside the container worker and write local checkpoints."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(
        min_pitch=min_pitch, max_pitch=max_pitch, w=w, latent_dim=latent_dim, chord_dim=0
    ).to(device)
    conditioner = Conditioner(min_pitch=min_pitch, max_pitch=max_pitch, w=w).to(device)
    discriminator = Discriminator(
        min_pitch=min_pitch, max_pitch=max_pitch, w=w, chord_dim=0
    ).to(device)
    
    if fine_tuned:
        checkpoint_path = f"{LOCAL_CHECKPOINT_DIR}/{genre}/model_weights_latest.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        if download_weight_from_gcs(
            f"checkpoints/{genre}/model_weights_latest.pth", 
            checkpoint_path
        ): 
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint["generator_state_dict"])
            conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        else: 
            raise FileNotFoundError(f"No pre-trained weights found for genre '{genre}' at gs://{MODEL_WEIGHTS_BUCKET}/checkpoints/{genre}/model_weights_latest.pth")
        
    dataset = DrumDataset(metadata_path=metadata_path, genre_filter=genre, device=device)
    if not dataset:
        raise ValueError(f"No data found for genre '{genre}'.")

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    if train_size == 0 or eval_size == 0:
        raise ValueError("The dataset must contain enough bars for train and validation splits.")
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42)
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers
    )

    generator_optimizer = torch.optim.RMSprop(
        itertools.chain(generator.parameters(), conditioner.parameters()), lr=learning_rate
    )
    discriminator_optimizer = torch.optim.RMSprop(
        discriminator.parameters(), lr=learning_rate
    )

    options = {
        "epochs": epochs,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "genre": genre,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "w": w,
        "latent_dim": latent_dim,
    }

    checkpoints = train(
        generator,
        conditioner,
        discriminator,
        train_dataloader,
        eval_dataloader,
        generator_optimizer,
        discriminator_optimizer,
        options,
        device
    )
    
    datestamp = time.strftime("%Y%m%d-%H%M%S")
    for checkpoint in checkpoints:
        upload_weight_to_gcs(
            checkpoint,
            f"checkpoints/{genre}/model_weights_{datestamp}.pth"
        )
    if not any(Path(checkpoint_dir).glob(f"{genre}*.pth")):
        raise FileNotFoundError("Training completed without producing a checkpoint.")


def stage_dataset(stage_dir: Path) -> Path:
    """Download metadata.json and referenced MIDI files from the training data bucket using MIDI_PREFIX."""
    client = storage.Client()
    training_bucket = client.bucket(TRAINING_DATA_BUCKET)

    metadata_path = stage_dir / "metadata.json"
    metadata_blob = training_bucket.blob("metadata.json")
    if not metadata_blob.exists():
        raise FileNotFoundError(f"metadata.json not found in gs://{TRAINING_DATA_BUCKET}/")
    
    metadata_blob.download_to_filename(metadata_path)

    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)

    for item in metadata.get("files", []):
        midi_path_raw = item["midi_path"]
        normalized_name = Path(midi_path_raw).name
        item["midi_path"] = normalized_name
        
        destination = stage_dir / normalized_name
        blob_name = f"{MIDI_PREFIX}/{normalized_name}" if MIDI_PREFIX else normalized_name
        midi_blob = training_bucket.blob(blob_name)
        
        if midi_blob.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            midi_blob.download_to_filename(destination)
        else:
            LOGGER.warning("MIDI file not found at gs://%s/%s", TRAINING_DATA_BUCKET, blob_name)

    with metadata_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file)
        
    return metadata_path


@app.route("/", methods=["GET"])
def health_check():
    """Cloud Run health check endpoint."""
    return jsonify({"status": "healthy", "service": "training-service"}), 200


@app.route("/train", methods=["POST"])
def trigger_training():
    """API endpoint triggered by Google Cloud Scheduler with TRAINING_API_KEY validation."""
    expected_api_key = os.getenv("TRAINING_API_KEY")
    if expected_api_key:
        auth_header = request.headers.get("X-Training-Api-Key", "")
        if auth_header != expected_api_key:
            return jsonify({"error": "Unauthorized API key"}), 401

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"status": "error", "message": "Missing or invalid JSON payload"}), 400

    required_keys = [
        "genre", "epochs", "batch_size", "learning_rate", 
        "latent_dim", "min_pitch", "max_pitch", "w", 
        "lambda_1", "lambda_2", "pin_memory", "fine_tune", "num_workers"
    ]

    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        return jsonify({
            "status": "error", 
            "message": f"Missing required payload parameters: {', '.join(missing_keys)}"
        }), 400

    try:
        genre = str(payload["genre"])
        epochs = int(payload["epochs"])
        batch_size = int(payload["batch_size"])
        learning_rate = float(payload["learning_rate"])
        latent_dim = int(payload["latent_dim"])
        min_pitch = int(payload["min_pitch"])
        max_pitch = int(payload["max_pitch"])
        w = int(payload["w"])
        lambda_1 = float(payload["lambda_1"])
        lambda_2 = float(payload["lambda_2"])
        
        pin_memory_raw = payload["pin_memory"]
        pin_memory = pin_memory_raw if isinstance(pin_memory_raw, bool) else str(pin_memory_raw).lower() == "true"

        fine_tune_raw = payload["fine_tune"]
        fine_tuned = fine_tune_raw if isinstance(fine_tune_raw, bool) else str(fine_tune_raw).lower() == "true"

        num_workers = int(payload["num_workers"])

    except (ValueError, TypeError) as e:
        return jsonify({
            "status": "error", 
            "message": f"Parameter type validation failed: {str(e)}"
        }), 400

    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stage_dir = Path(temporary_directory) / "dataset"
            metadata_path = stage_dataset(stage_dir)
            checkpoint_dir = Path(temporary_directory) / "checkpoints"
            
            run_training(
                metadata_path=str(metadata_path),
                checkpoint_dir=str(checkpoint_dir),
                genre=genre,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                latent_dim=latent_dim,
                min_pitch=min_pitch,
                max_pitch=max_pitch,
                w=w,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                pin_memory=pin_memory,
                num_workers=num_workers,
                fine_tuned=fine_tuned
            )
            
            LOGGER.info("Successfully completed training for genre '%s' and uploaded checkpoints to bucket %s", genre, MODEL_WEIGHTS_BUCKET)
            
        return jsonify({
            "status": "success", 
            "message": "Training completed successfully", 
            "genre": genre,
            "weights_bucket": MODEL_WEIGHTS_BUCKET
        }), 200

    except Exception as e:
        LOGGER.exception("Training job failed with error: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)