"""
Basic training script for corner detection model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from .config import TrainingConfig
from paper_detection.model.model import create_model
from paper_detection.model.train.dataset import create_dataloaders


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
    total_relative_error = 0
    num_batches = 0

    per_image_errors = []
    per_image_names = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            corners = batch['corners'].to(device)
            image_names = batch.get('image_name', [None]*images.shape[0])

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
            pixel_errors = distances.mean(dim=1).cpu().numpy()  # shape: [batch]
            relative_errors = (distances.mean(dim=1) / image_size * 100).cpu().numpy()

            # Store per-image errors and names
            per_image_errors.extend(pixel_errors.tolist())
            per_image_names.extend(image_names)

            pixel_error = pixel_errors.mean()
            relative_error = relative_errors.mean()

            total_pixel_error += pixel_error
            total_relative_error += relative_error
            num_batches += 1

    # Najdi nejlepší a 5 nejhorších obrázků
    error_info = list(zip(per_image_names, per_image_errors))
    error_info_sorted = sorted(error_info, key=lambda x: x[1])
    best_image = error_info_sorted[0] if error_info_sorted else None
    worst_images = error_info_sorted[-5:] if len(error_info_sorted) >= 5 else error_info_sorted[-len(error_info_sorted):]

    print("\nPer-image pixel error distribution:")
    if best_image:
        print(f"Best image: {best_image[0]} | pixel error: {best_image[1]:.2f} px")
    if worst_images:
        print("Worst 5 images:")
        for name, err in reversed(worst_images):
            print(f"  {name} | pixel error: {err:.2f} px")

    avg_loss = total_loss / num_batches
    avg_pixel_error = total_pixel_error / num_batches
    avg_relative_error = total_relative_error / num_batches

    return avg_loss, avg_pixel_error, avg_relative_error


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
    train_loader, val_loader = create_dataloaders()
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_model(device=config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,     # sníží LR na polovinu
        patience=10,    # když se 10 epoch val loss nezlepší
        min_lr=1e-6,    # minimální hranice
        verbose=True
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    best_relative_error = float('inf')

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)

        # Validate
        val_loss, val_pixel_error, val_relative_error = validate(model, val_loader, criterion, config.device, config.image_size)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val Pixel Error: {val_pixel_error:.2f}px ({val_relative_error:.2f}%)")

        # Scheduler step
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pixel_error = val_pixel_error
            best_relative_error = val_relative_error
            checkpoint_path = config.output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_pixel_error': val_pixel_error,
                'val_relative_error': val_relative_error,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f}, pixel_error: {val_pixel_error:.2f}px, relative_error: {val_relative_error:.2f}%)")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best pixel error: {best_pixel_error:.2f}px ({best_relative_error:.2f}%)")
    print(f"Model saved to: {config.output_dir / 'best_model.pth'}")
    print("=" * 70)


def main():
    """Main entry point for training"""
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


if __name__ == "__main__":
    main()
