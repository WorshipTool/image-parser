"""
CNN for detecting paper corners using pretrained ResNet18 backbone.

This module provides a corner detection model based on pretrained ResNet18,
designed to work well even with small datasets (~30 images) by leveraging
transfer learning from ImageNet.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CornerDetectorCNN(nn.Module):
    """
    Corner detection model using pretrained ResNet18 backbone.

    Uses transfer learning with a pretrained ResNet18 as feature extractor,
    followed by a regression head that outputs 8 normalized coordinates
    representing the four corners of a paper.

    Input:
        - RGB image tensor of shape [B, 3, H, W] (typically 224x224 or 256x256)
        - Images should be normalized with ImageNet statistics:
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    Output:
        - Tensor of shape [B, 8] containing normalized coordinates [0, 1]:
          [x1, y1, x2, y2, x3, y3, x4, y4]
        - Coordinates represent the four corners of the detected paper
        - Values are in range [0, 1] and need to be scaled to image dimensions

    Architecture:
        1. ResNet18 backbone (pretrained on ImageNet)
           - Removes the final FC layer
           - Outputs 512-dimensional feature vector
        2. Regression head:
           - Linear(512, 256) + ReLU + Dropout(0.5)
           - Linear(256, 128) + ReLU + Dropout(0.3)
           - Linear(128, 8) + Sigmoid

    Training strategy:
        - Option 1 (default): Freeze backbone, train only regression head
        - Option 2: Fine-tune entire network with low learning rate
        - Use freeze_backbone() and unfreeze_backbone() methods to control

    Args:
        pretrained (bool): Use pretrained ResNet18 weights. Default: True
        freeze_backbone (bool): Freeze ResNet18 backbone during training. Default: True
        dropout (float): Dropout probability for regression head. Default: 0.5

    Example:
        >>> # Training from scratch on small dataset
        >>> model = CornerDetectorCNN(pretrained=True, freeze_backbone=True)
        >>> # Train only the regression head first
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>>
        >>> # After some epochs, optionally unfreeze and fine-tune
        >>> model.unfreeze_backbone()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    """

    def __init__(self, pretrained=True, freeze_backbone=True, dropout=0.5):
        super(CornerDetectorCNN, self).__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # Remove the final FC layer (we'll add our own regression head)
        # ResNet18 outputs 512 features after avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

        # Regression head for corner prediction
        # ResNet18 outputs 512 features, we need 8 outputs (4 corners × 2 coords)
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.6),  # Slightly less dropout in second layer
            nn.Linear(128, 8),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def freeze_backbone(self):
        """Freeze all parameters in the ResNet18 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters in the ResNet18 backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n=1):
        """
        Unfreeze the last N residual blocks of ResNet18 for gradual fine-tuning.

        ResNet18 has 4 layer groups (layer1, layer2, layer3, layer4).
        This method unfreezes the last n groups.

        Args:
            n (int): Number of layer groups to unfreeze (1-4). Default: 1
        """
        # ResNet18 structure: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        # We want to unfreeze last n "layerX" groups

        layers = []
        for module in self.backbone.children():
            if isinstance(module, nn.Sequential):  # layer1, layer2, layer3, layer4
                layers.append(module)

        # Unfreeze last n layers
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W]

        Returns:
            torch.Tensor: Corner coordinates of shape [B, 8] with values in [0, 1]
        """
        # Extract features with ResNet18 backbone
        features = self.backbone(x)  # [B, 512, 1, 1]

        # Predict corners with regression head
        corners = self.regression_head(features)  # [B, 8]

        return corners

    def get_trainable_parameters(self):
        """
        Get count of trainable vs total parameters.

        Returns:
            tuple: (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# Keep old model for backwards compatibility (if someone has old checkpoints)
class CornerDetectorCNN_Legacy(nn.Module):
    """
    DEPRECATED: Lightweight CNN without pretrained weights.

    This is the old model architecture kept for backwards compatibility.
    Use CornerDetectorCNN with pretrained ResNet18 for better performance.
    """

    def __init__(self, dropout=0.5):
        super(CornerDetectorCNN_Legacy, self).__init__()

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
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.adaptive_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x
