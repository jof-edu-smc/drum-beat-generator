"""Vertex AI batch-training entrypoint for the drumbeat GAN."""

from __future__ import annotations

import argparse
import json
import time
import itertools
import logging
import os
import random
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np # type: ignore
import torch # type: ignore
from google.cloud import storage # type: ignore
from torch.utils.data import DataLoader # type: ignore

from datasets.dataset import DrumDataset
from models.gan import Conditioner, Discriminator, Generator
from utilities.train import train

LOGGER = logging.getLogger(__name__)

# Configuration from environment variables
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-gan-model-bucket")
LOCAL_CHECKPOINT_DIR = "/tmp/checkpoints"
os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)

def upload_weight_to_gcs(local_path, destination_blob_name):
    """Uploads a trained model checkpoint to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Successfully uploaded {local_path} to gs://{BUCKET_NAME}/{destination_blob_name}")

def download_weight_from_gcs(source_blob_name, local_path):
    """Downloads an existing model checkpoint from GCS for version resumption."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_blob_name)
    if blob.exists():
        blob.download_to_filename(local_path)
        print(f"Successfully downloaded gs://{BUCKET_NAME}/{source_blob_name} to {local_path}")
        return True
    print(f"No existing checkpoint found at gs://{BUCKET_NAME}/{source_blob_name}. Starting fresh.")
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
    fine_tuned: bool = False,
) -> None:
    """Train the GAN inside a Vertex worker and write local checkpoints."""
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
        # Download the pre-trained weights from GCS
        if download_weight_from_gcs(
            f"checkpoints/{genre}/model_weights_latest.pth", 
            checkpoint_path
        ): 
            # Load the pre-trained weights into the model
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint["generator_state_dict"])
            conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        else: 
            raise FileNotFoundError(f"No pre-trained weights found for genre '{genre}' at gs://{BUCKET_NAME}/checkpoints/{genre}/model_weights_latest.pth")
        
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
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

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


def _gcs_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"Expected a gs:// URI, received: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_blob(client: storage.Client, uri: str, destination: Path) -> None:
    bucket_name, blob_name = _gcs_parts(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(destination)


def _relative_dataset_path(path: str) -> str:
    normalized = Path(path)
    parts = normalized.parts
    if "datasets" in parts:
        return str(Path(*parts[parts.index("datasets") + 1 :]))
    return normalized.name


def stage_dataset(metadata_uri: str, dataset_uri: str, stage_dir: Path) -> Path:
    """Download metadata and every MIDI referenced by it into a local tree."""
    client = storage.Client()
    metadata_path = stage_dir / "metadata.json"
    _download_blob(client, metadata_uri, metadata_path)

    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)

    dataset_bucket, dataset_prefix = _gcs_parts(dataset_uri)
    dataset_prefix = dataset_prefix.rstrip("/")
    for item in metadata.get("files", []):
        relative_path = _relative_dataset_path(item["midi_path"])
        item["midi_path"] = relative_path
        destination = stage_dir / relative_path
        source_uri = f"gs://{dataset_bucket}/{dataset_prefix}/{relative_path}"
        _download_blob(client, source_uri, destination)

    with metadata_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file)
    return metadata_path


def upload_directory(directory: Path, output_uri: str) -> None:
    client = storage.Client()
    bucket_name, prefix = _gcs_parts(output_uri)
    bucket = client.bucket(bucket_name)
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(directory).as_posix()
            bucket.blob(f"{prefix.rstrip('/')}/{relative_path}").upload_from_filename(file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-uri", required=True, help="gs:// URI for metadata.json")
    parser.add_argument("--dataset-uri", required=True, help="gs:// prefix containing MIDI files")
    parser.add_argument("--output-uri", default=os.getenv("AIP_MODEL_DIR"))
    parser.add_argument("--genre", default=os.getenv("GENRE", "all"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("N_EPOCHS", "10")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "32")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.0002")))
    parser.add_argument("--latent-dim", type=int, default=int(os.getenv("LATENT_DIM", "100")))
    parser.add_argument("--min-pitch", type=int, default=int(os.getenv("MIN_PITCH", "22")))
    parser.add_argument("--max-pitch", type=int, default=int(os.getenv("MAX_PITCH", "59")))
    parser.add_argument("--w", type=int, default=int(os.getenv("W", "16")))
    parser.add_argument("--lambda-1", type=float, default=float(os.getenv("LAMBDA_1", "0.01")))
    parser.add_argument("--lambda-2", type=float, default=float(os.getenv("LAMBDA_2", "0.1")))
    parser.add_argument("--fine-tuned", action="store_true", help="Whether to fine-tune an existing model")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if not args.output_uri:
        raise ValueError("--output-uri or AIP_MODEL_DIR is required")

    with tempfile.TemporaryDirectory() as temporary_directory:
        stage_dir = Path(temporary_directory) / "dataset"
        metadata_path = stage_dataset(args.metadata_uri, args.dataset_uri, stage_dir)
        checkpoint_dir = Path(temporary_directory) / "checkpoints"
        run_training(
            metadata_path=str(metadata_path),
            checkpoint_dir=str(checkpoint_dir),
            genre=args.genre,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            latent_dim=args.latent_dim,
            min_pitch=args.min_pitch,
            max_pitch=args.max_pitch,
            w=args.w,
            lambda_1=args.lambda_1,
            lambda_2=args.lambda_2,
            fine_tuned=args.fine_tuned
        )
        upload_directory(checkpoint_dir, args.output_uri)
        LOGGER.info("Uploaded checkpoints to %s", args.output_uri)


if __name__ == "__main__":
    main()
