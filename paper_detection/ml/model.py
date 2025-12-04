"""
Lightweight CNN for detecting paper corners in images.

This module provides a simple convolutional neural network designed for small
datasets (~31 training images). The architecture is intentionally kept small
to avoid overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CornerDetectorCNN(nn.Module):
    """
    Lightweight CNN for detecting four corners of a paper in an image.

    The network is designed to work with small datasets and outputs normalized
    corner coordinates. The architecture uses gradually increasing channels with
    aggressive pooling to keep the parameter count low.

    Input:
        - RGB image tensor of shape [B, 3, 224, 224]
        - Images should be normalized with ImageNet statistics:
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    Output:
        - Tensor of shape [B, 8] containing normalized coordinates [0, 1]:
          [x1, y1, x2, y2, x3, y3, x4, y4]
        - Coordinates represent the four corners of the detected paper
        - Values are in range [0, 1] and can be scaled to image dimensions

    Architecture:
        The network uses 4 convolutional blocks followed by fully connected layers:

        Conv Block 1: [B, 3, 224, 224] -> [B, 32, 56, 56]
            - Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
            - BatchNorm2d(32)
            - ReLU
            - MaxPool2d(2, 2)

        Conv Block 2: [B, 32, 56, 56] -> [B, 64, 14, 14]
            - Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
            - BatchNorm2d(64)
            - ReLU
            - MaxPool2d(2, 2)

        Conv Block 3: [B, 64, 14, 14] -> [B, 128, 7, 7]
            - Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            - BatchNorm2d(128)
            - ReLU
            - MaxPool2d(2, 2)

        Conv Block 4: [B, 128, 7, 7] -> [B, 256, 1, 1]
            - Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            - BatchNorm2d(256)
            - ReLU
            - AdaptiveAvgPool2d((1, 1))

        Fully Connected: [B, 256] -> [B, 8]
            - Flatten
            - Dropout(p=0.5)
            - Linear(256, 128)
            - ReLU
            - Dropout(p=0.3)
            - Linear(128, 8)
            - Sigmoid (outputs in [0, 1] range)

    Note:
        The architecture is intentionally small to avoid overfitting on a small
        dataset of ~31 images. Data augmentation is strongly recommended during
        training.

    TODO: When more data available (>100 images), consider using pretrained
          ResNet18 backbone for better feature extraction and generalization.

    Args:
        dropout (float): Dropout probability for fully connected layers.
                        Default: 0.5

    Example:
        >>> model = CornerDetectorCNN(dropout=0.5)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> corners = model(image)
        >>> print(corners.shape)  # torch.Size([1, 8])
    """

    def __init__(self, dropout=0.5):
        super(CornerDetectorCNN, self).__init__()

        # Conv Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Conv Block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, 224, 224]

        Returns:
            torch.Tensor: Corner coordinates of shape [B, 8] with values in [0, 1]
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.adaptive_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x
