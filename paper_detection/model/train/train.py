"""
Basic training script for corner detection model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from .config import TrainingConfig
from .model import create_model
from .dataset import create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        images = batch['image'].to(device)
        corners = batch['corners'].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)

        # Compute loss
        loss = criterion(predictions, corners)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device, image_size=224):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_pixel_error = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            corners = batch['corners'].to(device)

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss = criterion(predictions, corners)
            total_loss += loss.item()

            # Compute pixel error (denormalize and measure distance)
            pred_pixels = predictions * image_size
            true_pixels = corners * image_size

            # Reshape to [batch, 4, 2]
            pred_pixels = pred_pixels.view(-1, 4, 2)
            true_pixels = true_pixels.view(-1, 4, 2)

            # Euclidean distance for each corner
            distances = torch.sqrt(torch.sum((pred_pixels - true_pixels) ** 2, dim=2))
            pixel_error = distances.mean().item()

            total_pixel_error += pixel_error
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_pixel_error = total_pixel_error / num_batches

    return avg_loss, avg_pixel_error


def train(config: TrainingConfig = None):
    """Main training function"""

    if config is None:
        config = TrainingConfig()

    print("=" * 70)
    print("Paper Corner Detection - Training")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Image size: {config.image_size}")
    print("=" * 70)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(train_split=config.train_split)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_model(num_corners=config.num_corners, device=config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_pixel_error = float('inf')

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)

        # Validate
        val_loss, val_pixel_error = validate(model, val_loader, criterion, config.device, config.image_size)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Pixel Error: {val_pixel_error:.2f}px")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pixel_error = val_pixel_error
            checkpoint_path = config.output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_pixel_error': val_pixel_error,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f}, pixel_error: {val_pixel_error:.2f}px)")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best pixel error: {best_pixel_error:.2f}px")
    print(f"Model saved to: {config.output_dir / 'best_model.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    # Create config
    config = TrainingConfig()

    # Auto-detect CUDA
    if torch.cuda.is_available():
        config.device = "cuda"
        print("CUDA available! Using GPU")
    else:
        config.device = "cpu"
        print("CUDA not available, using CPU")

    # Run training
    train(config)
