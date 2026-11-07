import os
import torch # type: ignore
import wandb # type: ignore
import time
from tqdm import tqdm # type: ignore
import time
from typing import Any
from .losses import standard_gan_loss, feature_matching_loss

def train(
    generator, conditioner, discriminator,
    train_dataloader, val_dataloader,
    generator_optimizer, discriminator_optimizer,
    opt, device, checkpoint_dir='checkpoints'
):
    

    
    best_val_g_fm = float('inf')
    train_start_time = time.time()
    epochs = opt.epochs

    if len(train_dataloader) == 0:
        raise ValueError("Training dataloader is empty; no training batches were produced.")
    if len(val_dataloader) == 0:
        raise ValueError("Validation dataloader is empty; no validation batches were produced.")

    wandb_init = getattr(wandb, "init", None)
    if not callable(wandb_init):
        raise RuntimeError("The installed wandb package does not provide wandb.init().")

    run: Any = wandb_init(
        entity="jof3",
        project="drumbeat-generator",
        mode=os.getenv("WANDB_MODE", "offline"),
        config={
            "learning_rate": opt.lr,
            "architecture": "GAN with Conditioner",
            "dataset": "Apple Loops",
            "epochs": epochs,
        },
    )

    D_loss_standard = torch.zeros((), device=device)
    G_loss_bce = torch.zeros((), device=device)
    G_loss_fm = torch.zeros((), device=device)
 
    for epoch_num in range(epochs):
        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_num + 1}/{epochs}", unit="batch", leave=False)
        
        previous_bars = {} 
         
        for item in batch_pbar:
            real_bar = item["bar"].to(device=device, dtype=torch.float32)
            genres = item["genre"]

            condition_bar = torch.zeros_like(real_bar)

            for index, genre in enumerate(genres):
                if genre in previous_bars:
                    condition_bar[index] = previous_bars[genre]
                else:
                    condition_bar[index] = 0

            for index, genre in enumerate(genres):
                previous_bars[genre] = real_bar[index].detach()
            batch_size = real_bar.size(0)

            conditioned_bar = to_conditioner_input(condition_bar)
            chord_condition = torch.zeros(batch_size, 13, device=device, dtype=torch.float32)

            pitch_slice = slice(opt.min_pitch, opt.max_pitch + 1)

            real_bar = real_bar[:, :, pitch_slice, :]
            conditioned_bar = conditioned_bar[:, :, pitch_slice, :]
            # --- Update Discriminator ---
            discriminator_optimizer.zero_grad()

            with torch.no_grad():
                # --- Generate fake data ---
                z_noise = torch.randn(batch_size, opt.latent_dim, device=device)
                c4, c3, c2, c1 = conditioner(conditioned_bar)
                fake_bar = generator(z_noise, c1=c1, c2=c2, c3=c3, c4=c4, chords=chord_condition)  # Pass chord conditioning if desired

            # --- Discriminator forward pass ---
            D_out_real, _ = discriminator(real_bar, chord_condition=chord_condition, return_features=True)
            D_out_fake, _ = discriminator(fake_bar.detach(), chord_condition=chord_condition, return_features=True)  # Detach to avoid backprop through Generator

            # --- Calculate losses ---
            D_loss_standard, G_loss_bce = standard_gan_loss(D_out_real, D_out_fake)
            D_loss_standard.backward()  # Backprop BCE loss for Discriminator
            discriminator_optimizer.step()
            with torch.no_grad():
                D_out_real_target, f_real_target = discriminator(real_bar, chord_condition=chord_condition, return_features=True)

            # --- Update Generator Twice ---
            for _ in range(2):
                generator_optimizer.zero_grad()

                # Generate fresh fake data
                z = torch.randn(batch_size, opt.latent_dim, device=device)
                c4, c3, c2, c1 = conditioner(conditioned_bar)
                fake_data = generator(z, c1=c1, c2=c2, c3=c3, c4=c4, chords=chord_condition)

                # 2. Only pass the FAKE data through the Discriminator inside the loop
                D_out_fake, f_fake = discriminator(fake_data, chord_condition=chord_condition, return_features=True)

                # 3. Use the detached targets for the loss functions
                _, G_loss_bce = standard_gan_loss(D_out_real_target, D_out_fake)

                G_loss_fm = feature_matching_loss(
                    real_bar.view(batch_size, -1),
                    fake_data.view(batch_size, -1),
                    f_real_target.view(batch_size, -1), # Use the static target
                    f_fake.view(batch_size, -1),
                    lambda_1=opt.lambda_1,
                    lambda_2=opt.lambda_2
                )

                # Combine losses into a single total loss before backprop
                G_total_loss = G_loss_bce + G_loss_fm
                G_total_loss.backward()  # Backprop combined loss for Generator
                generator_optimizer.step()

            batch_pbar.set_postfix({
                "D_loss": f"{D_loss_standard.item():.4f}",
                "G_bce": f"{G_loss_bce.item():.4f}",
                "G_fm": f"{G_loss_fm.item():.4f}"
            })

            
        generator.eval()
        discriminator.eval()
        conditioner.eval()

        val_d_loss = 0.0
        val_g_bce = 0.0
        val_g_fm = 0.0
        val_steps = 0

        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch_num + 1}/{epochs}", unit="batch", leave=False)
            pitch_slice = slice(opt.min_pitch, opt.max_pitch + 1)
            validation_previous_bars = {}
            for val_item in val_pbar:
                val_real_bar = val_item["bar"].to(device=device, dtype=torch.float32)
                val_genres = val_item["genre"]
                val_condition_bar = torch.zeros_like(val_real_bar)

                for index, genre in enumerate(val_genres):
                    if genre in validation_previous_bars:
                        val_condition_bar[index] = validation_previous_bars[genre]

                for index, genre in enumerate(val_genres):
                    validation_previous_bars[genre] = val_real_bar[index].detach()

                val_conditioned_bar = to_conditioner_input(val_condition_bar)
                val_bsz = val_real_bar.size(0)

                # Match training preprocessing
                val_real_bar = val_real_bar[:, :, pitch_slice, :]
                val_conditioned_bar = val_conditioned_bar[:, :, pitch_slice, :]

                # Build zero chord per actual batch size, not opt.batch_size
                chord_condition = torch.zeros(val_bsz, 13, device=device, dtype=torch.float32)

                z_val = torch.randn(val_bsz, opt.latent_dim, device=device)
                vc4, vc3, vc2, vc1 = conditioner(val_conditioned_bar)
                val_fake_bar = generator(z_val, c1=vc1, c2=vc2, c3=vc3, c4=vc4, chords=chord_condition)

                val_D_out_real, val_f_real = discriminator(val_real_bar, chord_condition=chord_condition, return_features=True)
                val_D_out_fake, val_f_fake = discriminator(val_fake_bar, chord_condition=chord_condition, return_features=True)

                d_loss, g_bce = standard_gan_loss(val_D_out_real, val_D_out_fake)
                g_fm = feature_matching_loss(
                    val_real_bar.view(val_bsz, -1),
                    val_fake_bar.view(val_bsz, -1),
                    val_f_real.view(val_bsz, -1),
                    val_f_fake.view(val_bsz, -1),
                    lambda_1=opt.lambda_1, #0.1,
                    lambda_2=opt.lambda_2 #1.0
                )

                val_d_loss += d_loss.item()
                val_g_bce += g_bce.item()
                val_g_fm += g_fm.item()
                val_steps += 1

            val_d_loss /= max(1, val_steps)
            val_g_bce /= max(1, val_steps)
            val_g_fm /= max(1, val_steps)

        print(f"Epoch {epoch_num + 1} completed \n    --> Validation losses - D: {val_d_loss:.4f}, G BCE: {val_g_bce:.4f}, G FM: {val_g_fm:.4f}")
        generator.train()
        discriminator.train()
        conditioner.train()

        checkpoint = {
                "epoch": epoch_num,
                "generator_state_dict": generator.state_dict(),
                "conditioner_state_dict": conditioner.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                "train_D_loss": float(D_loss_standard.item()),
                "train_G_bce": float(G_loss_bce.item()),
                "train_G_fm": float(G_loss_fm.item()),
                "val_D_loss": float(val_d_loss),
                "val_G_bce": float(val_g_bce),
                "val_G_fm": float(val_g_fm),
                "config": {
                    "latent_dim": opt.latent_dim,
                    "batch_size": opt.batch_size,
                    "lr": opt.lr,
                    "min_pitch": opt.min_pitch,
                    "max_pitch": opt.max_pitch,
                    "w": opt.w,
                },
            }
        fm_tag = f"{int(round(val_g_fm * 1_000_000)):06d}"
        should_save = (epoch_num > 0 and epoch_num % 5 == 0) or (epoch_num > 0 and val_g_fm < best_val_g_fm)

        if should_save:
            datestamp = time.strftime("%Y%m%d-%H%M%S")
            print(f"New best model found at epoch {epoch_num + 1} with val G_fm: {val_g_fm:.4f}. Saving checkpoint at {datestamp}.")
            if val_g_fm < best_val_g_fm:
                best_val_g_fm = val_g_fm
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"{opt.genre}_e{epoch_num + 1}_{fm_tag}_{datestamp}.pth"))


        # --- Logging ---
        print(f"Discriminator Loss: {D_loss_standard.item():.4f}")
        print(f"Generator BCE Loss: {G_loss_bce.item():.4f}")
        print(f"Generator Feature Matching Loss: {G_loss_fm.item():.4f}")
        
        run.log({
            "loss/discriminator": float(D_loss_standard.item()),
            "loss/generator": float(G_loss_bce.item()),
            "loss/feature_matching": float(G_loss_fm.item()),
            "epoch": epoch_num + 1,
            "batch": batch_pbar.n
        })

    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    run.finish()
    print(f"Total training time: {total_train_time:.2f} seconds ({total_train_time / 60:.2f} minutes)")

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
