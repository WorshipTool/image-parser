"""
U-Net model for paper segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: Upsample -> Concat -> DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder (to be upsampled)
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch between x1 and x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for paper segmentation

    Architecture:
    - Encoder: 4 downsampling blocks
    - Bottleneck: center block
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 1-channel mask logits
    """

    def __init__(self, in_channels=3, out_channels=1, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Mask logits [B, 1, H, W]
        """
        # Encoder
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 512, H/16, W/16] (or 1024 if not bilinear)

        # Decoder with skip connections
        x = self.up1(x5, x4)  # [B, 256, H/8, W/8]
        x = self.up2(x, x3)   # [B, 128, H/4, W/4]
        x = self.up3(x, x2)   # [B, 64, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]

        # Output layer
        logits = self.outc(x)  # [B, 1, H, W]
        return logits


def create_unet(device="cpu"):
    """
    Create and initialize U-Net model

    Args:
        device: Device to move model to ('cpu' or 'cuda')

    Returns:
        U-Net model
    """
    model = UNet(in_channels=3, out_channels=1, bilinear=True)
    model = model.to(device)
    return model


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation

    Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice loss = 1 - Dice coefficient
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]

        Returns:
            Dice loss (scalar)
        """
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)

        # Flatten for easier computation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss

    This combines the strengths of both losses:
    - BCE: Good for pixel-wise classification
    - Dice: Good for handling class imbalance
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]

        Returns:
            Combined loss (scalar)
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
