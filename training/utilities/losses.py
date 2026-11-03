import torch
from torch import nn

# MODEL ARCHITECTURE 
def standard_gan_loss(D_out_real, D_out_fake):
    """Standard Minimax BCE Loss for GANs"""
    criterion = nn.BCELoss()

    # Create target tensors filled with 1s (real) and 0s (fake)
    real_targets = torch.ones_like(D_out_real)
    fake_targets = torch.zeros_like(D_out_fake)

    # Discriminator loss: How well did it classify real as 1 and fake as 0?
    D_loss_real = criterion(D_out_real, real_targets)
    D_loss_fake = criterion(D_out_fake, fake_targets)
    D_loss = D_loss_real + D_loss_fake

    # Generator loss: How well did it trick the Discriminator into guessing 1?
    G_loss_bce = criterion(D_out_fake, real_targets)

    return D_loss, G_loss_bce

def feature_matching_loss(real_data, fake_data, f_real, f_fake, lambda_1=0.1, lambda_2=1.0):
    """
    Implements Equation 2 from the MidiNet paper:
    lambda_1 || E[X] - E[G(z)] ||^2_2 + lambda_2 || E[f(X)] - E[f(G(z))] ||^2_2
    """
    # Calculate Expectations (mean across the batch dimension)
    mean_real = torch.mean(real_data, dim=0)
    mean_fake = torch.mean(fake_data, dim=0)

    mean_f_real = torch.mean(f_real, dim=0)
    mean_f_fake = torch.mean(f_fake, dim=0)

    # Calculate Squared L2 Norms
    l2_raw_data = torch.sum((mean_real - mean_fake) ** 2)
    l2_features = torch.sum((mean_f_real - mean_f_fake) ** 2)
    
    return (lambda_1 * l2_raw_data) + (lambda_2 * l2_features)
    