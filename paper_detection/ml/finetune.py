"""
Phase 2 Fine-Tuning Script for Paper Corner Detection

This script performs fine-tuning of a pretrained ResNet18-based corner detector
by unfreezing the last ResNet blocks and training with a lower learning rate.

Usage:
    python3 -m paper_detection.ml.finetune --unfreeze_blocks 1 --epochs 50 --lr 0.0001

Args:
    --unfreeze_blocks: Number of ResNet blocks to unfreeze (1-4). Default: 1
    --epochs: Number of fine-tuning epochs. Default: 50
    --lr: Learning rate for fine-tuning. Default: 1e-4
    --checkpoint: Path to Phase 1 checkpoint. Default: paper_detection/models/paper_detector_cnn.pth
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper_detection.ml.model import CornerDetectorCNN
from paper_detection.ml.dataset import PaperCornersDataset
from paper_detection.ml.config import TrainingConfig
from paper_detection.ml.augmentations import get_training_augmentation, get_validation_augmentation


def calculate_pixel_error(predictions, targets, image_size=256):
    """
    Calculate average pixel error for corner predictions.

    Args:
        predictions: Predicted normalized corners (batch_size, 8)
        targets: Ground truth normalized corners (batch_size, 8)
        image_size: Image size for denormalization

    Returns:
        Average pixel error in pixels
    """
    # Denormalize to pixel coordinates
    pred_pixels = predictions * image_size
    target_pixels = targets * image_size

    # Calculate Euclidean distance for each corner
    # Reshape to (batch_size, 4, 2) for 4 corners with (x, y)
    pred_corners = pred_pixels.view(-1, 4, 2)
    target_corners = target_pixels.view(-1, 4, 2)

    # Calculate L2 distance for each corner
    distances = torch.sqrt(torch.sum((pred_corners - target_corners) ** 2, dim=2))

    # Average across all corners and batch
    return distances.mean().item()


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        corners = batch["corners"].to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, corners)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, config):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_pixel_error = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            corners = batch["corners"].to(device)

            predictions = model(images)
            loss = criterion(predictions, corners)
            pixel_error = calculate_pixel_error(predictions, corners, config.image_size)

            total_loss += loss.item()
            total_pixel_error += pixel_error
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "px_error": f"{pixel_error:.2f}"})

    avg_loss = total_loss / len(dataloader)
    avg_pixel_error = total_pixel_error / len(dataloader)

    return avg_loss, avg_pixel_error


def main():
    parser = argparse.ArgumentParser(description="Fine-tune paper corner detection model")
    parser.add_argument("--unfreeze_blocks", type=int, default=1,
                        help="Number of ResNet blocks to unfreeze (1-4)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--checkpoint", type=str,
                        default="paper_detection/models/paper_detector_cnn.pth",
                        help="Path to Phase 1 checkpoint")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for fine-tuning")
    args = parser.parse_args()

    # Load config (mostly from TrainingConfig defaults)
    config = TrainingConfig()
    config.learning_rate = args.lr
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size

    print("=" * 70)
    print("PAPER CORNER DETECTION - PHASE 2 FINE-TUNING")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Unfreezing last {args.unfreeze_blocks} ResNet block(s)")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)
    print()

    device = torch.device(config.device)
    print(f"Using device: {device}\n")

    # Load datasets
    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    train_dataset = PaperCornersDataset(
        images_dir="paper_detection/tests/test_images",
        ground_truth_path="paper_detection/tests/test_corners_ground_truth.json",
        config=config,
        mode="train"
    )

    val_dataset = PaperCornersDataset(
        images_dir="paper_detection/tests/test_images",
        ground_truth_path="paper_detection/tests/test_corners_ground_truth.json",
        config=config,
        mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Load model from checkpoint
    print("=" * 70)
    print("LOADING MODEL FROM CHECKPOINT")
    print("=" * 70)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Please run Phase 1 training first.")
        sys.exit(1)

    model = CornerDetectorCNN(
        pretrained=True,
        freeze_backbone=True,  # Initially frozen
        dropout=config.dropout
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    phase1_best_error = checkpoint.get('best_val_pixel_error', checkpoint.get('best_val_loss', 'unknown'))
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Phase 1 best pixel error: {phase1_best_error}")
    print()

    # Unfreeze last N blocks
    print("=" * 70)
    print("UNFREEZING BACKBONE LAYERS")
    print("=" * 70)
    model.unfreeze_last_n_blocks(args.unfreeze_blocks)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print()

    # Setup training
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        verbose=True
    )

    # Training loop
    print("=" * 70)
    print("STARTING FINE-TUNING")
    print("=" * 70)
    print(f"Total epochs: {config.num_epochs}")
    print(f"Initial learning rate: {config.learning_rate}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    print("=" * 70)
    print()

    best_val_pixel_error = float('inf')
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pixel_error": [],
        "learning_rates": [],
        "phase1_best_pixel_error": phase1_best_error
    }

    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)

        # Validate
        val_loss, val_pixel_error = validate(model, val_loader, criterion, device, config)

        # Update scheduler
        scheduler.step(val_pixel_error)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pixel_error"].append(val_pixel_error)
        history["learning_rates"].append(current_lr)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Pixel Error: {val_pixel_error:.2f}px | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Check if best model
        if val_pixel_error < best_val_pixel_error:
            best_val_pixel_error = val_pixel_error
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best model
            checkpoint_path = os.path.join(config.save_dir, "paper_detector_cnn_finetuned.pth")
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_pixel_error': best_val_pixel_error,
                'phase1_best_pixel_error': phase1_best_error,
                'unfrozen_blocks': args.unfreeze_blocks,
                'config': config
            }, checkpoint_path)

            print(f"  → Best model saved! (Val Pixel Error: {val_pixel_error:.2f}px)")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"No improvement in pixel error for {config.early_stopping_patience} epochs")
            break

        print()

    total_time = time.time() - start_time

    # Save history
    history_path = os.path.join(config.save_dir, "finetuning_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Print summary
    print("=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Total fine-tuning time: {total_time / 60:.2f} minutes")
    print(f"Phase 1 best pixel error: {phase1_best_error}")
    print(f"Phase 2 best pixel error: {best_val_pixel_error:.2f}px")

    if isinstance(phase1_best_error, (int, float)):
        improvement = ((phase1_best_error - best_val_pixel_error) / phase1_best_error) * 100
        print(f"Improvement: {improvement:.1f}%")

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Unfrozen blocks: {args.unfreeze_blocks}")
    print(f"Final learning rate: {current_lr:.6f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")
    print("=" * 70)

    if best_val_pixel_error < 40:
        print("\n🎉 SUCCESS! Target pixel error <40px achieved!")
    else:
        print(f"\n⚠️  Target not yet reached. Current: {best_val_pixel_error:.2f}px, Target: <40px")
        print(f"   Consider unfreezing more blocks or training longer.")


if __name__ == "__main__":
    main()
