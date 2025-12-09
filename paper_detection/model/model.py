"""
ResNet18-based model for paper corner detection
"""

import torch
import torch.nn as nn
from torchvision import models

from paper_detection.model.config import NUM_CORNERS


class CornerDetectionCNN(nn.Module):
    """ResNet18-based model for detecting 4 paper corners"""

    def __init__(self, pretrained=True):
        super().__init__()

        self.num_corners = NUM_CORNERS

        # Load pretrained ResNet18
        # Use weights parameter for newer PyTorch versions
        if pretrained:
            try:
                # Try new API (PyTorch >= 1.13)
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except (AttributeError, TypeError):
                # Fall back to old API
                resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet18(weights=None)

        # Use ResNet18 as feature extractor (remove final FC layer)
        # ResNet18 outputs 512 features after avgpool
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
            resnet.layer3,  # 256 channels
            resnet.layer4,  # 512 channels
            resnet.avgpool  # Global average pooling
        )

        # Regression head for corner coordinates
        # Input: 512 features from ResNet18
        # Output: 4 corners * 2 coordinates (x, y) = 8 values
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_corners * 2),  # 4 corners, 2 coords each
        )

    def forward(self, x):
        """
        Args:
            x: Input images [batch_size, 3, 224, 224]

        Returns:
            Corner coordinates [batch_size, 8] (normalized 0-1)
        """
        features = self.features(x)
        corners = self.regressor(features)
        return corners


def create_model(device="cpu"):
    """Create and initialize model"""
    model = CornerDetectionCNN()
    model = model.to(device)
    return model
