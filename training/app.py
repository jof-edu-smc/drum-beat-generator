import os
import random
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.gan import Generator, Conditioner, Discriminator
from training.datasets.dataset import GrooveDataset
from training.utilities.train import train



generator = Generator(min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w, latent_dim=opt.latent_dim, chord_dim=0)
conditioner = Conditioner(min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w)
discriminator = Discriminator(min_pitch=opt.min_pitch, max_pitch=opt.max_pitch, w=opt.w, chord_dim=0)

generator.to(device)
conditioner.to(device)
discriminator.to(device)

if TRAINING_ON:
    print(f"Initializing dataset and dataloaders with batch size {opt.batch_size} for genre type: {opt.genre}...")
    dataset = GrooveDataset(data_dir='./groove_dataset', pt_dir=f'./processed_tensors/{opt.genre}', genre_filter=opt.genre, device='cpu', include_apple_loops=True, apple_loops_dir='./test_midi')
    train_dataset = dataset.with_split('train')
    eval_dataset = dataset.with_split('eval')

    print(f"Training on: {device}")

    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    ckpt_dir = opt.ckpt_dir

    torch.manual_seed(seed)
    if not os.path.exists(dataset.pt_train):
        raise ValueError("Dataset not properly initialized. pt_train is None.")
    if device.type == "cuda":
        loader_kwargs = {
            "num_workers": 0,
            "pin_memory": opt.pin_memory
        }
    else:
        loader_kwargs = {
            "num_workers": 0,
            "pin_memory": False
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        **loader_kwargs
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        **loader_kwargs
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    g_c_params = itertools.chain(generator.parameters(), conditioner.parameters())
    generator_optimizer = torch.optim.RMSprop(g_c_params, lr=opt.lr)
    discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

    pretrained_path = os.path.join(opt.pretrained, f"{opt.genre}_model_weights.pth")
    if os.path.isfile(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    elif not os.path.isfile(pretrained_path):
        print("No Pretrained Model exists for this genre, init weights randomly")
        # Fine-tuning should reuse pretrained weights, so do not reinitialize them here.
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        conditioner.apply(weights_init_normal)

    generator.train()
    discriminator.train()
    conditioner.train()

    print(f"Batch Info: {opt.batch_size} batch size, {len(train_dataloader)} batches per epoch")

    train(
        generator,
        conditioner,
        discriminator,
        train_dataloader,
        eval_dataloader,
        generator_optimizer,
        discriminator_optimizer,
        opt,
        device,
        checkpoint_dir=ckpt_dir
    )

    possible_models = [f for f in os.listdir(ckpt_dir) if f.startswith(f"best_{opt.genre}_model_epoch_")]
    if not possible_models:
        raise FileExistsError("No saved models found in checkpoints directory.")
    else:
        latest_model = random.choice(possible_models)
        print(f"Training completed. Of all models: {possible_models}, loading the best one: {latest_model}")
        checkpoint = torch.load(os.path.join(ckpt_dir, latest_model), map_location=device)

if not TRAINING_ON and os.path.exists(opt.ckpt_dir):
    print("Training skipped. To enable training, set TRAINING_ON = True at the top of the notebook.")
    files = os.listdir(opt.ckpt_dir)
    checkpoint_files = [f for f in files if f.startswith(f"best_{opt.genre}_model_epoch_")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {opt.ckpt_dir}. Please ensure you have trained the model or placed a checkpoint in the directory.")
    else:
        latest_checkpoint = random.choice(checkpoint_files)
        print(f"Loading the latest checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(os.path.join(opt.ckpt_dir, latest_checkpoint), map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        print(f"Checkpoint loaded successfully from epoch {checkpoint['epoch'] + 1}. Ready for inference or further training.")

elif not TRAINING_ON and not os.path.exists(opt.ckpt_dir) and os.path.exists(opt.pretrained):
    print("Training skipped. To enable training, set TRAINING_ON = True at the top of the notebook.")
    files = os.listdir(opt.pretrained)
    checkpoint_files = [f for f in files if f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {opt.pretrained}. Please ensure you have trained the model or placed a checkpoint in the directory.")
    else:
        latest_checkpoint = f"{opt.genre}_model_weights.pth"
        print(f"Loading the latest checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(os.path.join(opt.pretrained, latest_checkpoint), map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        conditioner.load_state_dict(checkpoint["conditioner_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        print(f"Checkpoint loaded successfully from epoch {checkpoint['epoch'] + 1}. Ready for inference or further training.")