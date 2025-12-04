"""
Comprehensive training script for paper corner detection model.

This script trains a CNN model to detect four corners of a paper in images.
It handles dataset loading, training loop, validation, early stopping, and
model checkpointing.

Usage:
    # Train with default parameters
    python -m paper_detection.ml.train

    # Train with custom parameters
    python -m paper_detection.ml.train --epochs 300 --batch_size 16 --lr 0.001

    # Specify custom data paths
    python -m paper_detection.ml.train \
        --images_dir paper_detection/tests/test_images \
        --ground_truth_path paper_detection/tests/test_corners_ground_truth.json

Features:
    - Automatic train/validation split
    - Data augmentation (geometric + photometric)
    - Early stopping to prevent overfitting
    - Learning rate scheduling
    - Model checkpointing (saves best model)
    - Training history logging to JSON
    - Progress bars with tqdm (if available)
    - Comprehensive metrics (loss + pixel error)

Output:
    - Best model saved to: paper_detection/models/paper_detector_cnn.pth
    - Training history saved to: paper_detection/models/training_history.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
import argparse
import sys
from typing import Tuple

from .config import TrainingConfig
from .model import CornerDetectorCNN
from .dataset import PaperCornersDataset
from .augmentations import get_training_augmentation, get_inference_transform

# Try to import tqdm for progress bars (optional)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not available, progress bars will not be shown")


def compute_pixel_error(pred_corners: torch.Tensor, true_corners: torch.Tensor, image_size: int = 224) -> float:
    """
    Compute average pixel error between predicted and ground truth corners.

    This metric denormalizes the corner coordinates from [0, 1] range to pixel
    coordinates and computes the Euclidean distance for each corner pair.

    Args:
        pred_corners: Predicted corner coordinates of shape [batch_size, 8]
                     in normalized [0, 1] range
        true_corners: Ground truth corner coordinates of shape [batch_size, 8]
                     in normalized [0, 1] range
        image_size: Image dimensions for denormalization. Default: 224

    Returns:
        Average pixel distance across all corners in the batch

    Example:
        >>> pred = torch.tensor([[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]])
        >>> true = torch.tensor([[0.15, 0.15, 0.85, 0.15, 0.85, 0.85, 0.15, 0.85]])
        >>> error = compute_pixel_error(pred, true, image_size=224)
        >>> print(f"Average error: {error:.2f} pixels")
    """
    # Denormalize coordinates to pixels
    pred_pixels = pred_corners * image_size
    true_pixels = true_corners * image_size

    # Reshape from [batch_size, 8] to [batch_size, 4, 2]
    pred_pixels = pred_pixels.view(-1, 4, 2)
    true_pixels = true_pixels.view(-1, 4, 2)

    # Compute Euclidean distance for each corner
    distances = torch.sqrt(torch.sum((pred_pixels - true_pixels) ** 2, dim=2))

    # Return average distance across all corners in batch
    return distances.mean().item()


def train_epoch(
    model: CornerDetectorCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.

    This function performs a complete pass through the training dataset,
    computing loss, backpropagating gradients, and updating model weights.

    Args:
        model: The neural network model to train
        dataloader: DataLoader providing training batches
        criterion: Loss function (e.g., SmoothL1Loss)
        optimizer: Optimizer for updating weights (e.g., AdamW)
        device: Device to run training on (cpu or cuda)

    Returns:
        Average training loss across all batches

    Note:
        The model is set to training mode (dropout enabled, batch norm in
        training mode) during this function.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    # Create iterator with or without tqdm
    if TQDM_AVAILABLE:
        iterator = tqdm(dataloader, desc="Training", leave=False)
    else:
        iterator = dataloader

    for batch in iterator:
        # Move data to device
        images = batch["image"].to(device)
        corners = batch["corners"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss = criterion(predictions, corners)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Update progress bar if available
        if TQDM_AVAILABLE:
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    # Return average loss
    return total_loss / num_batches


def validate_epoch(
    model: CornerDetectorCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    image_size: int = 224
) -> Tuple[float, float]:
    """
    Validate the model on validation dataset.

    This function evaluates the model on the validation set without updating
    weights. It computes both the loss and average pixel error metric.

    Args:
        model: The neural network model to validate
        dataloader: DataLoader providing validation batches
        criterion: Loss function (e.g., SmoothL1Loss)
        device: Device to run validation on (cpu or cuda)
        image_size: Image size for denormalizing coordinates. Default: 224

    Returns:
        Tuple of (average_loss, average_pixel_error)
        - average_loss: Mean loss across all validation samples
        - average_pixel_error: Mean pixel distance for corner predictions

    Note:
        The model is set to evaluation mode (dropout disabled, batch norm in
        eval mode) and gradients are disabled during this function.
    """
    model.eval()
    total_loss = 0.0
    total_pixel_error = 0.0
    num_batches = len(dataloader)

    # Create iterator with or without tqdm
    if TQDM_AVAILABLE:
        iterator = tqdm(dataloader, desc="Validation", leave=False)
    else:
        iterator = dataloader

    with torch.no_grad():
        for batch in iterator:
            # Move data to device
            images = batch["image"].to(device)
            corners = batch["corners"].to(device)

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss = criterion(predictions, corners)
            total_loss += loss.item()

            # Compute pixel error (denormalize corners first)
            pixel_error = compute_pixel_error(predictions, corners, image_size)
            total_pixel_error += pixel_error

            # Update progress bar if available
            if TQDM_AVAILABLE:
                iterator.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "px_error": f"{pixel_error:.2f}"
                })

    # Return average metrics
    avg_loss = total_loss / num_batches
    avg_pixel_error = total_pixel_error / num_batches

    return avg_loss, avg_pixel_error


def train_model(
    config: TrainingConfig,
    images_dir: str,
    ground_truth_path: str
) -> None:
    """
    Main training function that orchestrates the complete training workflow.

    This function handles:
    1. Dataset creation and splitting (train/val)
    2. Model initialization
    3. Optimizer and scheduler setup
    4. Training loop with validation
    5. Early stopping
    6. Model checkpointing
    7. Training history logging

    Args:
        config: TrainingConfig object with all hyperparameters
        images_dir: Path to directory containing training images
        ground_truth_path: Path to JSON file with ground truth corner annotations

    Training Flow:
        - For each epoch:
            1. Train on training set
            2. Validate on validation set
            3. Update learning rate if validation loss plateaus
            4. Save model if validation loss improves
            5. Stop early if no improvement for patience epochs
        - Save final training history to JSON

    Output Files:
        - {config.save_dir}/paper_detector_cnn.pth: Best model checkpoint
        - {config.save_dir}/training_history.json: Training metrics log

    Note:
        The function creates the save directory if it doesn't exist.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create save directory if it doesn't exist
    os.makedirs(config.save_dir, exist_ok=True)

    # Set device
    device = torch.device(config.device)
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create datasets
    print("\n" + "=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    train_dataset = PaperCornersDataset(
        images_dir=images_dir,
        ground_truth_path=ground_truth_path,
        config=config,
        mode="train"
        # Augmentation automatically selected based on mode
    )

    val_dataset = PaperCornersDataset(
        images_dir=images_dir,
        ground_truth_path=ground_truth_path,
        config=config,
        mode="val"
        # Validation transform automatically selected
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging, increase for faster training
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    model = CornerDetectorCNN(dropout=config.dropout)
    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model architecture: {model.__class__.__name__}")

    # Initialize optimizer (AdamW with weight decay for regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Initialize loss function (SmoothL1Loss is more robust to outliers than MSE)
    criterion = nn.SmoothL1Loss()

    # Initialize learning rate scheduler (reduce LR on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        verbose=True
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pixel_error": [],
        "learning_rate": [],
        "epochs": []
    }

    # Best model tracking
    best_val_loss = float("inf")
    best_val_pixel_error = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    # TODO: Add TensorBoard logging for better visualization
    # tensorboard_writer = SummaryWriter(log_dir=os.path.join(config.save_dir, "logs"))

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Total epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Initial learning rate: {config.learning_rate}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    print("=" * 70 + "\n")

    # Training loop
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_pixel_error = validate_epoch(
            model, val_loader, criterion, device, config.image_size
        )

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pixel_error"].append(val_pixel_error)
        history["learning_rate"].append(current_lr)
        history["epochs"].append(epoch)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print progress
        print(f"Epoch {epoch:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Pixel Error: {val_pixel_error:.2f}px | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Check if validation pixel error improved
        if val_pixel_error < best_val_pixel_error:
            best_val_pixel_error = val_pixel_error
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save best model
            model_path = os.path.join(config.save_dir, "paper_detector_cnn.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_pixel_error": val_pixel_error,
                "best_val_pixel_error": best_val_pixel_error,
                "config": config
            }, model_path)

            print(f"  → Best model saved! (Val Pixel Error: {val_pixel_error:.2f}px)")

            # TODO: Add checkpoint saving every N epochs
            # if epoch % 10 == 0:
            #     checkpoint_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch}.pth")
            #     torch.save({...}, checkpoint_path)

        else:
            epochs_without_improvement += 1

            # Check early stopping
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"No improvement in pixel error for {config.early_stopping_patience} epochs")
                break

        # TODO: Add gradient clipping if training becomes unstable
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"Best validation pixel error: {best_val_pixel_error:.2f}px (epoch {best_epoch})")
    print(f"Final learning rate: {current_lr:.6f}")
    print(f"Model saved to: {os.path.join(config.save_dir, 'paper_detector_cnn.pth')}")

    # Save training history to JSON
    history_path = os.path.join(config.save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")
    print("=" * 70 + "\n")


def main():
    """
    Main entry point for training script with command-line argument parsing.

    This function parses command-line arguments, creates a TrainingConfig,
    and initiates the training process.

    Command-line Arguments:
        --images_dir: Path to directory containing training images
                     Default: paper_detection/tests/test_images
        --ground_truth_path: Path to JSON file with corner annotations
                            Default: paper_detection/tests/test_corners_ground_truth.json
        --epochs: Number of training epochs
                 Default: 200
        --batch_size: Batch size for training
                     Default: 8
        --lr: Initial learning rate
             Default: 0.001

    Example:
        python -m paper_detection.ml.train --epochs 300 --batch_size 16
    """
    parser = argparse.ArgumentParser(
        description="Train paper corner detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        default="paper_detection/tests/test_images",
        help="Path to directory containing training images"
    )

    parser.add_argument(
        "--ground_truth_path",
        type=str,
        default="paper_detection/tests/test_corners_ground_truth.json",
        help="Path to JSON file with ground truth corner annotations"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate"
    )

    args = parser.parse_args()

    # Create training configuration from arguments
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Verify paths exist
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        sys.exit(1)

    if not os.path.exists(args.ground_truth_path):
        print(f"Error: Ground truth file not found: {args.ground_truth_path}")
        sys.exit(1)

    # Start training
    print("\n" + "=" * 70)
    print("PAPER CORNER DETECTION - TRAINING SCRIPT")
    print("=" * 70)
    print(f"Images directory: {args.images_dir}")
    print(f"Ground truth file: {args.ground_truth_path}")
    print(f"Configuration: {config}")
    print("=" * 70)

    train_model(config, args.images_dir, args.ground_truth_path)


if __name__ == "__main__":
    main()
