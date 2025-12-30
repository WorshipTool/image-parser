"""
Unit tests for U-Net model
"""

import pytest
import torch

from paper_detection.model.model import (
    UNet,
    create_unet,
    DiceLoss,
    CombinedLoss,
    DoubleConv,
    Down,
    Up
)


class TestUNetArchitecture:
    """Test U-Net model architecture"""

    def test_unet_creation(self):
        """Test basic U-Net creation"""
        model = UNet(in_channels=3, out_channels=1)
        assert model is not None
        assert model.in_channels == 3
        assert model.out_channels == 1

    def test_unet_forward_shape(self):
        """Test U-Net forward pass output shape"""
        model = UNet(in_channels=3, out_channels=1)
        batch_size = 2
        h, w = 384, 384

        # Create dummy input
        x = torch.randn(batch_size, 3, h, w)

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (batch_size, 1, h, w)

    def test_unet_different_sizes(self):
        """Test U-Net with different input sizes"""
        model = UNet(in_channels=3, out_channels=1)

        for size in [256, 384, 512]:
            x = torch.randn(1, 3, size, size)
            output = model(x)
            assert output.shape == (1, 1, size, size)

    def test_create_unet_cpu(self):
        """Test create_unet helper function"""
        model = create_unet(device="cpu")
        assert model is not None
        assert next(model.parameters()).device.type == "cpu"


class TestModelComponents:
    """Test individual model components"""

    def test_double_conv(self):
        """Test DoubleConv block"""
        block = DoubleConv(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        assert output.shape == (2, 128, 32, 32)

    def test_down_block(self):
        """Test Down block"""
        block = Down(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        # MaxPool2d reduces size by 2
        assert output.shape == (2, 128, 16, 16)

    def test_up_block(self):
        """Test Up block"""
        block = Up(in_channels=256, out_channels=64, bilinear=True)
        x1 = torch.randn(2, 128, 16, 16)  # From decoder
        x2 = torch.randn(2, 128, 32, 32)  # Skip connection from encoder
        output = block(x1, x2)
        assert output.shape == (2, 64, 32, 32)


class TestLossFunctions:
    """Test loss functions"""

    def test_dice_loss_perfect(self):
        """Test Dice loss with perfect prediction"""
        loss_fn = DiceLoss()

        # Perfect prediction (after sigmoid, probs = targets)
        logits = torch.ones(2, 1, 32, 32) * 10  # High logits → sigmoid ≈ 1
        targets = torch.ones(2, 1, 32, 32)

        loss = loss_fn(logits, targets)

        # Loss should be close to 0 for perfect prediction
        assert loss < 0.1

    def test_dice_loss_worst(self):
        """Test Dice loss with worst prediction"""
        loss_fn = DiceLoss()

        # Worst prediction (opposite)
        logits = torch.ones(2, 1, 32, 32) * 10   # sigmoid ≈ 1
        targets = torch.zeros(2, 1, 32, 32)       # all zeros

        loss = loss_fn(logits, targets)

        # Loss should be close to 1 for worst prediction
        assert loss > 0.9

    def test_combined_loss(self):
        """Test combined BCE + Dice loss"""
        loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

        loss = loss_fn(logits, targets)

        # Loss should be positive
        assert loss > 0
        assert torch.isfinite(loss)

    def test_combined_loss_weights(self):
        """Test that loss weights affect the result"""
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

        # Different weight configurations
        loss_fn_bce = CombinedLoss(bce_weight=1.0, dice_weight=0.0)
        loss_fn_dice = CombinedLoss(bce_weight=0.0, dice_weight=1.0)
        loss_fn_equal = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

        loss_bce = loss_fn_bce(logits, targets)
        loss_dice = loss_fn_dice(logits, targets)
        loss_equal = loss_fn_equal(logits, targets)

        # All losses should be positive and finite
        assert torch.isfinite(loss_bce)
        assert torch.isfinite(loss_dice)
        assert torch.isfinite(loss_equal)


class TestModelGradients:
    """Test model gradient flow"""

    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = UNet(in_channels=3, out_channels=1)
        criterion = CombinedLoss()

        # Create dummy input and target
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        target = torch.randint(0, 2, (1, 1, 256, 256)).float()

        # Forward pass
        output = model(x)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
