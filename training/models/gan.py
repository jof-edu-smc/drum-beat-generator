import torch
from torch import nn

def weights_init_normal(m):
    """
    Applies a zero-centered normal distribution (std=0.02) to the weights
    of Convolutional, Transposed Convolutional, and Linear layers.
    """
    classname = m.__class__.__name__

    # Check if the layer is a Conv, ConvTranspose, or Linear layer
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # Initialize weights to mean=0.0, std=0.02
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

        # If the layer has a bias, initialize it to 0
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    # For Batch Normalization layers, it is standard DCGAN practice
    # to center the weights at 1.0 rather than 0.0, with the same std.
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, chord_dim=13, min_pitch=48, max_pitch=72, w=16):
        super(Generator, self).__init__()

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.w = w
        self.pitch_slice = slice(min_pitch, max_pitch + 1)
        self.pitch_height = max_pitch - min_pitch + 1

        # --- NEW: Architectural Toggle ---
        self.chord_dim = chord_dim
        self.use_chords = chord_dim > 0

        # Only initialize chord projection layers if we are actually using chords
        if self.use_chords:
            self.chord_proj_step1 = nn.Linear(chord_dim, 1 * 1 * 2)
            self.chord_proj_step2 = nn.Linear(chord_dim, 1 * 1 * 4)
            self.chord_proj_step3 = nn.Linear(chord_dim, 1 * 1 * 8)
            self.chord_proj_step4 = nn.Linear(chord_dim, 1 * 1 * 16)

        # FC1 dynamically adjusts its input size
        self.fc1 = nn.Linear(latent_dim + chord_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)

        # --- CHANNEL MATH ---
        # Base channels: 256 (from x) + 256 (from conditions) = 512
        # If chords are used, we project them into 1 extra channel = 513
        transconv_in_channels = 513 if self.use_chords else 512

        self.transconv1 = nn.ConvTranspose2d(transconv_in_channels, 256, kernel_size=(1, 2), stride=(1, 2))
        self.transconv2 = nn.ConvTranspose2d(transconv_in_channels, 256, kernel_size=(1, 2), stride=(1, 2))
        self.transconv3 = nn.ConvTranspose2d(transconv_in_channels, 256, kernel_size=(1, 2), stride=(1, 2))
        self.transconv4 = nn.ConvTranspose2d(transconv_in_channels, 1, kernel_size=(self.pitch_height, 1), stride=(1, 1))

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, c4=None, c3=None, c2=None, c1=None, chords=None):
        if c4 is None or c3 is None or c2 is None or c1 is None:
            raise ValueError("All four condition layers (c4, c3, c2, c1) must be provided.")

        batch_size = z.size(0)

        # --- Latent Combination ---
        if self.use_chords:
            if chords is None:
                raise ValueError("Generator was initialized with chord_dim > 0, but no chords were provided.")
            chords = chords.squeeze(-1) if chords.ndim > 2 else chords
            z_combined = torch.cat([z, chords], dim=1)
        else:
            z_combined = z

        x = self.relu(self.fc1(z_combined))
        x = self.relu(self.fc2(x))

        # Reshape to (batch, 256, 1, 2)
        x = x.view(batch_size, 256, 1, 2)

        # --- Step 1 ---
        if self.use_chords:
            proj_c1 = self.chord_proj_step1(chords).view(batch_size, 1, 1, 2)
            x = torch.cat([x, c4, proj_c1], dim=1)
        else:
            x = torch.cat([x, c4], dim=1)

        x = self.relu(self.bn1(self.transconv1(x)))

        # --- Step 2 ---
        if self.use_chords:
            proj_c2 = self.chord_proj_step2(chords).view(batch_size, 1, 1, 4)
            x = torch.cat([x, c3, proj_c2], dim=1)
        else:
            x = torch.cat([x, c3], dim=1)

        x = self.relu(self.bn2(self.transconv2(x)))

        # --- Step 3 ---
        if self.use_chords:
            proj_c3 = self.chord_proj_step3(chords).view(batch_size, 1, 1, 8)
            x = torch.cat([x, c2, proj_c3], dim=1)
        else:
            x = torch.cat([x, c2], dim=1)

        x = self.relu(self.bn3(self.transconv3(x)))

        # --- Step 4 ---
        if self.use_chords:
            proj_c4 = self.chord_proj_step4(chords).view(batch_size, 1, 1, 16)
            x = torch.cat([x, c1, proj_c4], dim=1)
        else:
            x = torch.cat([x, c1], dim=1)

        x = self.transconv4(x)

        x = self.sigmoid(x)
        return x.squeeze(1)


class Discriminator(nn.Module):
    def __init__(self, chord_dim=13, min_pitch=48, max_pitch=72, w=16):
        super(Discriminator, self).__init__()

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.w = w
        self.pitch_slice = slice(min_pitch, max_pitch + 1)
        self.pitch_height = max_pitch - min_pitch + 1

        self.chord_dim = chord_dim
        self.use_chords = chord_dim > 0

        # Input channels: 1 (for the piano roll) + 13 (if chords are used)
        conv1_in_channels = 1 + chord_dim

        # First conv: 14 filters, 128x2 kernel, stride 2
        self.conv1 = nn.Conv2d(conv1_in_channels, 14, kernel_size=(self.pitch_height, 2), stride=(1, 2), padding=0)

        # Second conv: 77 filters, 1x4 kernel, stride 2
        self.conv2 = nn.Conv2d(14, 77, kernel_size=(1, 4), stride=(1, 2), padding=0)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(77 * 1 * 3, 1024)
        self.fc2 = nn.Linear(1024, 1)

        self.batchnorm2 = nn.BatchNorm2d(77)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bar, chord_condition=None, return_features=False):
        if bar.shape[1] != 1:
            bar = bar.unsqueeze(1)

        batch_size = bar.size(0)

        if self.use_chords:
            if chord_condition is None:
                raise ValueError("Discriminator requires chord conditioning, but none was provided.")
            # Dynamically use self.chord_dim instead of hardcoded 13
            chords_expanded = chord_condition.view(batch_size, self.chord_dim, 1, 1).expand(-1, -1, self.pitch_height, 16)
            bar = torch.cat([bar, chords_expanded], dim=1)

        # We isolate the output of conv1 so we can use it for feature matching
        f1 = self.relu(self.conv1(bar))
        x = self.relu(self.conv2(f1))
        x = self.batchnorm2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(x))

        out = out.squeeze(1)

        if return_features:
            return out, f1
        return out

class Conditioner(nn.Module):
    def __init__(self, min_pitch=48, max_pitch=72, w=16):
        super(Conditioner, self).__init__()
        # The Conditioner uses standard Conv2d to shrink the matrix.
        # Input shape: (batch, 1, 128, 16)
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.w = w
        self.pitch_slice = slice(min_pitch, max_pitch + 1)
        self.pitch_height = max_pitch - min_pitch + 1
        # 1. Input is [batch, 1, pitch_height, w]
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(self.pitch_height, 1), stride=(1, 1))  # -> (batch, 256, 1, 16)

        # 2. Start halving the width.
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(1, 2), stride=(1, 2))  # -> (batch, 256, 1, 8)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(1, 2), stride=(1, 2))  # -> (batch, 256, 1, 4)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(1, 2), stride=(1, 2))  # -> (batch, 256, 1, 2)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # x is the 2D condition (e.g., the melody of the previous bar)
        # Ensure x has the channel dimension: (batch, 1, 128, 16)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        c1 = self.relu(self.bn1(self.conv1(x)))    # Shape: (..., 1, 16)
        c2 = self.relu(self.bn2(self.conv2(c1)))   # Shape: (..., 1, 8)
        c3 = self.relu(self.bn3(self.conv3(c2)))   # Shape: (..., 1, 4)

        # The deepest layer
        c4 = self.relu(self.conv4(c3))             # Shape: (..., 1, 2)

        # Return them in reverse order so they easily pair with the
        # Generator's transposed convolutions (which start small and grow large).
        return c4, c3, c2, c1

