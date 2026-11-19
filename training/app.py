"""Flask API service for the drumbeat GAN training pipeline using minimal environment configuration and API key auth."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import torch.multiprocessing as mp # type: ignore
from dotenv import load_dotenv # type: ignore

from utilities.training_gan import Opt
from utilities.training_pipeline import run_training, stage_dataset

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
load_dotenv()


# Only the 4 requested environment variables are read here
TRAINING_DATA_BUCKET = os.getenv("TRAINING_DATA_BUCKET", "your-default-training-data-bucket")
MODEL_WEIGHTS_BUCKET = os.getenv("MODEL_WEIGHTS_BUCKET", "your-default-model-weights-bucket")
WANDB_MODE = os.getenv("WANDB_MODE", "offline")
DEV = bool(os.getenv("DEV", "False"))

LOCAL_CHECKPOINT_DIR = f"/tmp/{TRAINING_DATA_BUCKET}"
os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)


def trigger_training():
    keys_for_training = {
        "genre": os.getenv("GENRE", ""), 
        "epochs": os.getenv("EPOCHS", ""), 
        "batch_size": os.getenv("BATCH_SIZE", ""),
        "learning_rate": os.getenv("LEARNING_RATE", ""), 
        "latent_dim": os.getenv("LATENT_DIM", ""), 
        "min_pitch": os.getenv("MIN_PITCH", ""), 
        "max_pitch": os.getenv("MAX_PITCH", ""), 
        "w": os.getenv("W", ""), 
        "num_workers": os.getenv("NUM_WORKERS", ""), 
        "lambda_1": os.getenv("LAMBDA_1", ""), 
        "lambda_2": os.getenv("LAMBDA_2", ""), 
        "pin_memory": os.getenv("PIN_MEMORY", ""), 
        "fine_tune": os.getenv("FINE_TUNE", ""), 
        "dataset": os.getenv("DATASET", ""),
    }
    required_keys = [
        "genre", "epochs", "batch_size", "learning_rate", 
        "latent_dim", "min_pitch", "max_pitch", "w", 
        "lambda_1", "lambda_2", "pin_memory", "fine_tune", "num_workers", "dataset"
    ]

    missing_keys = [key for key in required_keys if key not in keys_for_training or not keys_for_training[key]]
    if missing_keys:
        raise ValueError(f"Missing required keys in payload: {', '.join(missing_keys)}")
    
    opt = Opt()
    
    try:    
        opt.genre = keys_for_training["genre"].lower()
        opt.epochs = int(keys_for_training["epochs"])
        opt.batch_size = int(keys_for_training["batch_size"])
        opt.learning_rate = float(keys_for_training["learning_rate"])
        opt.latent_dim = int(keys_for_training["latent_dim"])
        opt.min_pitch = int(keys_for_training["min_pitch"])
        opt.max_pitch = int(keys_for_training["max_pitch"])
        opt.w = int(keys_for_training["w"])
        opt.lambda_1 = float(keys_for_training["lambda_1"])
        opt.lambda_2 = float(keys_for_training["lambda_2"])
        opt.num_workers = int(keys_for_training["num_workers"])
        opt.dataset = str(keys_for_training.get("dataset", ""))
        opt.bucket_name_training_data = TRAINING_DATA_BUCKET
        opt.bucket_name_model_weights = MODEL_WEIGHTS_BUCKET
        opt.dev = DEV
        opt.local_checkpoint_dir = LOCAL_CHECKPOINT_DIR
        opt.wandb_mode = WANDB_MODE
        
        pin_memory_raw = keys_for_training["pin_memory"]
        opt.pin_memory = pin_memory_raw if isinstance(pin_memory_raw, bool) else str(pin_memory_raw).lower() == "true"

        fine_tune_raw = keys_for_training["fine_tune"]
        opt.fine_tuned = fine_tune_raw if isinstance(fine_tune_raw, bool) else str(fine_tune_raw).lower() == "true"

        opt.num_workers = int(keys_for_training["num_workers"])

    except ValueError as e:
        raise ValueError(f"Invalid value in payload: {e}")
    except KeyError as e:
        raise KeyError(f"Missing key in payload: {e}")

    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stage_dir = Path(temporary_directory) / "dataset"
            opt.metadata_path = stage_dataset(stage_dir, dataset_prefix=opt.dataset, bucket_name=TRAINING_DATA_BUCKET)
            opt.checkpoint_dir = Path(temporary_directory) 
            run_training(opt=opt)
            LOGGER.info("Successfully completed training for genre '%s' and uploaded checkpoints to bucket %s", opt.genre, MODEL_WEIGHTS_BUCKET)
            
        LOGGER.info("Training job completed successfully for genre '%s'. Checkpoints saved locally at %s.", opt.genre, opt.checkpoint_dir)

    except Exception as e:
        LOGGER.exception("Training job failed with error: %s", str(e))
        raise RuntimeError(f"Training job failed with error: {str(e)}")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass 
    trigger_training()