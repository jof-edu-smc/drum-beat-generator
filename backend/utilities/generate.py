import os
import torch
import random 
import numpy as np

def to_conditioner_input(x):
    # target: [B, 1, 128, 16]
    if x.ndim == 3:  # [B, 128, 16]
        return x.unsqueeze(1)
    if x.ndim == 4:
        if x.shape[1] == 1:  # already [B,1,128,16]
            return x
        if x.shape[1] == 16 and x.shape[-2:] == (128, 16):
            # x is [B, context_len, H, W] -> use last context bar
            return x[:, -1, :, :].unsqueeze(1)
    raise ValueError(f"Unexpected conditioner input shape: {x.shape}")

def generate_music(generator, conditioner,
                   opt, device, NUM_BARS_TO_GENERATE=4,
                   CHORD_PROGRESSION=None, 
                   checkpoint_path=None, creativity=1.0, custom_conditions=None):
    torch.seed()
    random.seed()
    np.random.seed()

    print(f"Generating {NUM_BARS_TO_GENERATE} bars of music with creativity={creativity} on device {device}")
    if CHORD_PROGRESSION is not None and NUM_BARS_TO_GENERATE != len(CHORD_PROGRESSION):
        raise ValueError("Length of NUM_BARS_TO_GENERATE must match length of CHORD_PROGRESSION")

    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided when TRAINING_ON is True.")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    conditioner.load_state_dict(checkpoint["conditioner_state_dict"])

    max_attempts = max(10, NUM_BARS_TO_GENERATE * 20)
    attempts = 0
    best_bar = None
    best_active = -1

    generator.eval()
    conditioner.eval()

    generated_bars = []
    while len(generated_bars) < NUM_BARS_TO_GENERATE and attempts < max_attempts:
        pitch_slice = slice(opt.min_pitch, opt.max_pitch + 1)
        pitch_h = opt.max_pitch - opt.min_pitch + 1

        if custom_conditions is not None:
            # Use provided tensors directly
            idx = len(generated_bars)
            condition_bar = custom_conditions[idx].to(device=device, dtype=torch.float32)
            # If the tensor is already (1, 128, 16), slice it to the model's pitch range
            if condition_bar.shape[-2] == 128:
                condition_bar = condition_bar[:, pitch_slice, :]
        else:
            if len(generated_bars) == 0:
                condition_bar = torch.zeros((1, pitch_h, opt.w), device=device, dtype=torch.float32)
            else:
                condition_bar = generated_bars[-1][pitch_slice, :].unsqueeze(0).to(device=device, dtype=torch.float32)

        condition_bar = to_conditioner_input(condition_bar)
        bsz = condition_bar.size(0)
        z_noise = (torch.randn(1, opt.latent_dim) * creativity).to(device)
        c4, c3, c2, c1 = conditioner(condition_bar)
        chord_condition = torch.zeros((bsz, 13), device=device, dtype=torch.float32)

        fake_bar = generator(z_noise, c1=c1, c2=c2, c3=c3, c4=c4, chords=chord_condition)

        full_bar = torch.zeros((bsz, 128, opt.w), device=device, dtype=fake_bar.dtype)
        full_bar[:, pitch_slice, :] = fake_bar
        active = (full_bar[:, pitch_slice, :] > 0.35).sum().item()

        if active > best_active:
            best_active = active
            best_bar = full_bar.squeeze(0).detach().cpu()

        if active < 3:
            attempts += 1
            if attempts >= max_attempts - 1 and best_bar is not None:
                generated_bars.append(best_bar)
            continue

        attempts += 1
        generated_bars.append(full_bar.squeeze(0).detach().cpu())
        
    return generated_bars