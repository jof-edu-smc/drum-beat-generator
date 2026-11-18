import os 
import json
import random
import logging
import itertools
import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from pathlib import Path
from google.cloud import storage # type: ignore

from datasets.dataset import DrumDataset
from models.gan import Conditioner, Discriminator, Generator
from utilities.training_gan import train, Opt
from utilities.testing import get_safe_num_workers


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def upload_weight_to_gcs(local_path, destination_blob_name, bucket_name):
    """Uploads a trained model checkpoint to the model weights GCS bucket."""
    print(f"Uploading {local_path} to gs://{bucket_name}/{destination_blob_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    LOGGER.info("Successfully uploaded %s to gs://%s/%s", local_path, bucket_name, destination_blob_name)


def download_weight_from_gcs(source_blob_name, local_path, bucket_name):
    """Downloads an existing model checkpoint from the model weights GCS bucket for fine-tuning."""
    print(f"Attempting to download weights from gs://{bucket_name}/{source_blob_name} to {local_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    if blob.exists():
        blob.download_to_filename(local_path)
        LOGGER.info("Successfully downloaded gs://%s/%s to %s", bucket_name, source_blob_name, local_path)
        return True
    LOGGER.info("No existing checkpoint found at gs://%s/%s. Starting fresh.", bucket_name, source_blob_name)
    return False

def run_training(opt: Opt) -> None:
    """Train the GAN inside the container worker and write local checkpoints."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(
        min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w, latent_dim=opt.latent_dim, chord_dim=0
    ).to(device)
    conditioner = Conditioner(min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w).to(device)
    discriminator = Discriminator(
        min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w, chord_dim=0
    ).to(device)
    
    if opt.fine_tuned:
        print(f"Attempting to load pre-trained weights for genre '{opt.genre}' from gs://{opt.bucket_name_model_weights}/{opt.genre}/model_weights_latest.pth")
        checkpoint_path = f"{opt.local_checkpoint_dir}/{opt.genre}/model_weights_latest.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        if download_weight_from_gcs(
            f"{opt.genre}/model_weights_latest.pth", 
            checkpoint_path,
            opt.bucket_name_model_weights
        ): 
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint["generator_state_dict"])
            conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        else: 
            raise FileNotFoundError(f"No pre-trained weights found for genre '{opt.genre}' at gs://{opt.bucket_name_model_weights}/{opt.genre}/model_weights_latest.pth")
        
    dataset = DrumDataset(metadata_path=opt.metadata_path, genre_filter=opt.genre, device=device)
    if not dataset:
        raise ValueError(f"No data found for genre '{opt.genre}'.")

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    if train_size == 0 or eval_size == 0:
        raise ValueError("The dataset must contain enough bars for train and validation splits.")
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Training dataset size: {len(train_dataset)}, Evaluation dataset size: {len(eval_dataset)}")
    safe_num_workers = get_safe_num_workers(opt.num_workers)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=opt.pin_memory, num_workers=safe_num_workers
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=opt.pin_memory, num_workers=safe_num_workers
    )
    print(f"Training dataloader batches: {len(train_dataloader)}, Evaluation dataloader batches: {len(eval_dataloader)}")
    generator_optimizer = torch.optim.RMSprop(
        itertools.chain(generator.parameters(), conditioner.parameters()), lr=opt.learning_rate
    )
    discriminator_optimizer = torch.optim.RMSprop(
        discriminator.parameters(), lr=opt.learning_rate
    )
    
    checkpoints = train(
        generator,
        conditioner,
        discriminator,
        train_dataloader,
        eval_dataloader,
        generator_optimizer,
        discriminator_optimizer,
        opt,
        device
    )
    print(f"Training completed for genre '{opt.genre}'. Checkpoints saved locally at {opt.checkpoint_dir}.")
    
    # all_e21_101585_20261118-142115.pth 
    for checkpoint in checkpoints:
        ckpt_info = checkpoint.split("_") 
        # ckpt_genre = ckpt_info[0]
        ckpt_epoch = ckpt_info[1]
        ckpt_fm_tag = ckpt_info[2]
        ckpt_timestamp = ckpt_info[3].split(".")[0]
        upload_weight_to_gcs(
            checkpoint,
            f"{opt.genre}/model_weights_{ckpt_epoch}_{ckpt_fm_tag}_{ckpt_timestamp}.pth",
            opt.bucket_name_model_weights
        )
    if not any(Path(opt.checkpoint_dir).glob(f"{opt.genre}*.pth")):
        raise FileNotFoundError("Training completed without producing a checkpoint.")

def stage_dataset(stage_dir: Path, dataset_prefix: str, bucket_name: str) -> Path:
    """Download metadata.json and referenced MIDI files from the training data bucket using metadata.json as a guide, and stage them in a temporary directory for training."""
    client = storage.Client()
    training_bucket = client.bucket(bucket_name)

    metadata_path = stage_dir / dataset_prefix / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_blob = training_bucket.blob(f"{dataset_prefix}/metadata.json")
    if not metadata_blob.exists():
        raise FileNotFoundError(f"metadata.json not found in gs://{bucket_name}/{dataset_prefix}/metadata.json")
    
    metadata_blob.download_to_filename(metadata_path)
    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
        print(f"Downloaded metadata.json with {len(metadata.get('files', []))} entries from gs://{bucket_name}/{dataset_prefix}/metadata.json to {metadata_path}")
    
    for item in metadata.get("files", []):
        midi_path = item["midi_path"]
        destination = stage_dir / dataset_prefix / midi_path
        
        blob_name = f"{dataset_prefix}/{midi_path}" if dataset_prefix not in midi_path else midi_path
        
        midi_blob = training_bucket.blob(blob_name)
        
        if midi_blob.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            midi_blob.download_to_filename(destination)
        else:
            LOGGER.warning("MIDI file not found at gs://%s/%s", bucket_name, blob_name)
    print(f"Staged {len(metadata.get('files', []))} MIDI files to {stage_dir / dataset_prefix}")
    with metadata_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file)
        
    return metadata_path