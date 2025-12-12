"""
ResNet18-based model for paper corner detection
"""

import torch
import torch.nn as nn
from torchvision import models

from paper_detection.model.config import NUM_CORNERS, IMAGE_SIZE



class CornerDetectionCNN(nn.Module):
    """ResNet18-based model for detecting 4 paper corners"""

    def __init__(self, pretrained=True, freeze_backbone=False, dropout=0.3):
        super().__init__()

        freeze_backbone = True


        self.num_corners = NUM_CORNERS
        # Load pretrained ResNet18
        if pretrained:
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except (AttributeError, TypeError):
                resnet = models.resnet18(pretrained=True)
        else:
            try:
                resnet = models.resnet18(weights=None)
            except TypeError:
                resnet = models.resnet18(pretrained=False)
        # Use ResNet18 as backbone (remove final FC layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * IMAGE_SIZE * IMAGE_SIZE // (32 * 32), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_corners * 2)
        )

    def forward(self, x):
        feats = self.backbone(x)  # [B, 512, 1, 1]
        out = self.head(feats)    # [B, 8]
        return out


def create_model(device="cpu", pretrained=True, freeze_backbone=False, dropout=0.3):
    """Create and initialize model (API compatible, now supports pretrained/freeze_backbone/dropout)"""
    model = CornerDetectionCNN(pretrained=pretrained, freeze_backbone=freeze_backbone, dropout=dropout)
    model = model.to(device)
    return model
