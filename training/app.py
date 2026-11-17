"""Flask API service for the drumbeat GAN training pipeline using minimal environment configuration and API key auth."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np # type: ignore
from flask import Flask, jsonify, request # type: ignore
from dotenv import load_dotenv # type: ignore

from utilities.training_gan import Opt
from utilities.training_pipeline import run_training, stage_dataset

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)

# Only the 4 requested environment variables are read here
TRAINING_DATA_BUCKET = os.getenv("TRAINING_DATA_BUCKET", "your-default-training-data-bucket")
MODEL_WEIGHTS_BUCKET = os.getenv("MODEL_WEIGHTS_BUCKET", "your-default-model-weights-bucket")
WANDB_MODE = os.getenv("WANDB_MODE", "offline")
DEV = bool(os.getenv("DEV", "False"))

LOCAL_CHECKPOINT_DIR = f"/tmp/{TRAINING_DATA_BUCKET}"
os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)

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
        "lambda_1", "lambda_2", "pin_memory", "fine_tune", "num_workers", "dataset"
    ]

    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        return jsonify({
            "status": "error", 
            "message": f"Missing required payload parameters: {', '.join(missing_keys)}"
        }), 400

    try:
        opt = Opt()
        opt.genre = str(payload["genre"])
        opt.epochs = int(payload["epochs"])
        opt.batch_size = int(payload["batch_size"])
        opt.learning_rate = float(payload["learning_rate"])
        opt.latent_dim = int(payload["latent_dim"])
        opt.min_pitch = int(payload["min_pitch"])
        opt.max_pitch = int(payload["max_pitch"])
        opt.w = int(payload["w"])
        opt.lambda_1 = float(payload["lambda_1"])
        opt.lambda_2 = float(payload["lambda_2"])
        dataset_prefix = str(payload.get("dataset", ""))
        opt.bucket_name_training_data = TRAINING_DATA_BUCKET
        opt.bucket_name_model_weights = MODEL_WEIGHTS_BUCKET
        opt.dev = DEV
        opt.local_checkpoint_dir = LOCAL_CHECKPOINT_DIR
        opt.wandb_mode = WANDB_MODE
        
        pin_memory_raw = payload["pin_memory"]
        opt.pin_memory = pin_memory_raw if isinstance(pin_memory_raw, bool) else str(pin_memory_raw).lower() == "true"

        fine_tune_raw = payload["fine_tune"]
        opt.fine_tuned = fine_tune_raw if isinstance(fine_tune_raw, bool) else str(fine_tune_raw).lower() == "true"

        opt.num_workers = int(payload["num_workers"])

    except (ValueError, TypeError) as e:
        return jsonify({
            "status": "error", 
            "message": f"Parameter type validation failed: {str(e)}"
        }), 400

    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stage_dir = Path(temporary_directory) / "dataset"
            opt.metadata_path = stage_dataset(stage_dir, dataset_prefix=dataset_prefix, bucket_name=TRAINING_DATA_BUCKET)
            opt.checkpoint_dir = Path(temporary_directory) 
            run_training(opt=opt)
            LOGGER.info("Successfully completed training for genre '%s' and uploaded checkpoints to bucket %s", opt.genre, MODEL_WEIGHTS_BUCKET)
            
        return jsonify({
            "status": "success", 
            "message": "Training completed successfully", 
            "genre": opt.genre,
            "weights_bucket": MODEL_WEIGHTS_BUCKET
        }), 200

    except Exception as e:
        LOGGER.exception("Training job failed with error: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    if DEV:
        port = int(os.environ.get("PORT", 5002))
    else: 
        port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=DEV)