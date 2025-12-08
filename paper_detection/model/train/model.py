"""
Simple CNN model for paper corner detection
"""

import torch
import torch.nn as nn


class CornerDetectionCNN(nn.Module):
    """Simple CNN for detecting 4 paper corners"""

    def __init__(self, num_corners=4):
        super().__init__()

        self.num_corners = num_corners

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )

        # Regression head for corner coordinates
        # Output: 4 corners * 2 coordinates (x, y) = 8 values
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_corners * 2),  # 4 corners, 2 coords each
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


def create_model(num_corners=4, device="cpu"):
    """Create and initialize model"""
    model = CornerDetectionCNN(num_corners=num_corners)
    model = model.to(device)
    return model
