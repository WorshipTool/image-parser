"""
Training script for paper segmentation U-Net model
"""

import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from paper_detection.segmentation.config import SegmentationConfig
from paper_detection.segmentation.model import create_unet, CombinedLoss
from paper_detection.segmentation.dataset import create_dataloaders
from paper_detection.segmentation.postprocess import mask_to_corners, min_corner_matching_error


def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) metric

    Args:
        pred_mask: Predicted mask logits [B, 1, H, W]
        true_mask: Ground truth mask [B, 1, H, W]
        threshold: Threshold for binarization

    Returns:
        IoU score
    """
    # Apply sigmoid and threshold
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    true_binary = true_mask

    # Calculate intersection and union
    intersection = (pred_binary * true_binary).sum()
    union = pred_binary.sum() + true_binary.sum() - intersection

    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou.item()


def calculate_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Dice coefficient metric

    Args:
        pred_mask: Predicted mask logits [B, 1, H, W]
        true_mask: Ground truth mask [B, 1, H, W]
        threshold: Threshold for binarization

    Returns:
        Dice score
    """
    # Apply sigmoid and threshold
    pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    true_binary = true_mask

    # Calculate intersection
    intersection = (pred_binary * true_binary).sum()
    total = pred_binary.sum() + true_binary.sum()

    # Avoid division by zero
    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection) / total
    return dice.item()


def calculate_corner_error(
    pred_masks: torch.Tensor,
    gt_corners: torch.Tensor,
    config: SegmentationConfig
) -> tuple:
    """
    Calculate corner detection error from predicted masks

    Args:
        pred_masks: Predicted mask logits [B, 1, H, W]
        gt_corners: Ground truth corners [B, 4, 2] in pixel coordinates
        config: Configuration

    Returns:
        Tuple of (mean_error_pixels, mean_error_percent, success_rate)
        - mean_error_pixels: Average L2 distance in pixels
        - mean_error_percent: Error as percentage of image diagonal
        - success_rate: Percentage of images where corners were successfully extracted
    """
    batch_size = pred_masks.size(0)
    image_size = pred_masks.size(2)  # Assuming square images

    # Calculate diagonal for percentage calculation
    diagonal = np.sqrt(2 * image_size ** 2)

    errors_pixels = []
    successful_extractions = 0

    for i in range(batch_size):
        # Get predicted mask as numpy array
        pred_mask_prob = torch.sigmoid(pred_masks[i, 0]).cpu().numpy()  # [H, W]

        # Extract corners from mask
        pred_corners = mask_to_corners(
            pred_mask_prob,
            threshold=config.MASK_THRESHOLD,
            min_contour_area=config.MIN_CONTOUR_AREA,
            approx_epsilon=config.APPROX_EPSILON,
            debug=False
        )

        if pred_corners is not None:
            successful_extractions += 1

            # Get ground truth corners
            gt_corners_np = gt_corners[i].cpu().numpy()  # [4, 2]

            # Calculate minimal matching error (handles rotation and mirroring)
            error_pixels, _ = min_corner_matching_error(pred_corners, gt_corners_np)
            errors_pixels.append(error_pixels)

    # Calculate statistics
    if len(errors_pixels) > 0:
        mean_error_pixels = np.mean(errors_pixels)
        mean_error_percent = (mean_error_pixels / diagonal) * 100
    else:
        mean_error_pixels = float('inf')
        mean_error_percent = float('inf')

    success_rate = (successful_extractions / batch_size) * 100

    return mean_error_pixels, mean_error_percent, success_rate


def visualize_predictions(
    images: torch.Tensor,
    true_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    image_names: list,
    output_dir: Path,
    epoch: int,
    config: SegmentationConfig
):
    """
    Save visualization of predictions (mask + corners overlay)

    Args:
        images: Batch of images [B, 3, H, W]
        true_masks: Ground truth masks [B, 1, H, W]
        pred_masks: Predicted mask logits [B, 1, H, W]
        image_names: List of image names
        output_dir: Directory to save visualizations
        epoch: Current epoch number
        config: Configuration
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = min(images.size(0), config.NUM_VAL_VIS_SAMPLES)

    for i in range(batch_size):
        # Get tensors
        image_tensor = images[i]  # [3, H, W]
        true_mask_tensor = true_masks[i, 0]  # [H, W]
        pred_mask_tensor = pred_masks[i, 0]  # [H, W]

        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_denorm = image_tensor * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)

        # Convert to numpy
        image_np = (image_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        true_mask_np = (true_mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        pred_mask_prob = torch.sigmoid(pred_mask_tensor).cpu().numpy()
        pred_mask_np = (pred_mask_prob * 255).astype(np.uint8)

        # Extract corners from predicted mask
        pred_corners = mask_to_corners(
            pred_mask_prob,
            threshold=config.MASK_THRESHOLD,
            min_contour_area=config.MIN_CONTOUR_AREA,
            approx_epsilon=config.APPROX_EPSILON,
            debug=False
        )

        # Create visualization
        h, w = image_bgr.shape[:2]
        vis = np.zeros((h, w * 3, 3), dtype=np.uint8)

        # Column 1: Original image
        vis[:, :w] = image_bgr

        # Column 2: Ground truth mask overlay
        gt_overlay = image_bgr.copy()
        gt_mask_colored = cv2.applyColorMap(true_mask_np, cv2.COLORMAP_JET)
        gt_overlay = cv2.addWeighted(gt_overlay, 0.6, gt_mask_colored, 0.4, 0)
        vis[:, w:2*w] = gt_overlay

        # Column 3: Predicted mask overlay + corners
        pred_overlay = image_bgr.copy()
        pred_mask_colored = cv2.applyColorMap(pred_mask_np, cv2.COLORMAP_JET)
        pred_overlay = cv2.addWeighted(pred_overlay, 0.6, pred_mask_colored, 0.4, 0)

        # Draw predicted corners if found
        if pred_corners is not None:
            # Draw polygon
            corners_int = pred_corners.astype(np.int32)
            cv2.polylines(pred_overlay, [corners_int], True, (0, 255, 0), 2)

            # Draw corner points
            for corner in corners_int:
                cv2.circle(pred_overlay, tuple(corner), 5, (0, 255, 0), -1)

        vis[:, 2*w:] = pred_overlay

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        cv2.putText(vis, "Original", (10, 30), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(vis, "GT Mask", (w + 10, 30), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(vis, "Pred Mask + Corners", (2*w + 10, 30), font, font_scale, (255, 255, 255), font_thickness)

        # Save visualization
        output_path = output_dir / f"epoch_{epoch:03d}_{image_names[i]}"
        cv2.imwrite(str(output_path), vis)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)  # [B, 3, H, W]
        masks = batch['mask'].to(device)    # [B, 1, H, W]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # [B, 1, H, W]
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        iou = calculate_iou(outputs, masks)
        dice = calculate_dice(outputs, masks)

        # Accumulate
        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'iou': f"{iou:.4f}",
            'dice': f"{dice:.4f}"
        })

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches

    return avg_loss, avg_iou, avg_dice


def validate_epoch(model, val_loader, criterion, device, config, epoch, save_vis=False):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_corner_error_pixels = 0.0
    total_corner_error_percent = 0.0
    total_corner_success_rate = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            corners = batch['corners'].to(device)  # [B, 4, 2]
            image_names = batch['image_name']

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Calculate metrics
            iou = calculate_iou(outputs, masks)
            dice = calculate_dice(outputs, masks)

            # Calculate corner error
            corner_error_px, corner_error_pct, corner_success = calculate_corner_error(
                outputs, corners, config
            )

            # Accumulate
            total_loss += loss.item()
            total_iou += iou
            total_dice += dice
            total_corner_error_pixels += corner_error_px
            total_corner_error_percent += corner_error_pct
            total_corner_success_rate += corner_success
            num_batches += 1

            # Save visualization for first batch only
            if save_vis and batch_idx == 0:
                visualize_predictions(
                    images, masks, outputs, image_names,
                    config.DEBUG_DIR / "val_predictions",
                    epoch, config
                )

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    avg_corner_error_pixels = total_corner_error_pixels / num_batches
    avg_corner_error_percent = total_corner_error_percent / num_batches
    avg_corner_success_rate = total_corner_success_rate / num_batches

    return (avg_loss, avg_iou, avg_dice,
            avg_corner_error_pixels, avg_corner_error_percent, avg_corner_success_rate)


def train(resume_from: str = None):
    """
    Main training function

    Args:
        resume_from: Path to checkpoint to resume from, or special values:
                    - None: Start training from scratch
                    - 'latest': Resume from latest checkpoint
                    - 'best': Resume from best model
                    - '<path>': Resume from specific checkpoint file
    """
    # Load configuration
    config = SegmentationConfig()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.DEVICE = str(device)
    print(f"Using device: {device}")

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    print("Creating model...")
    model = create_unet(device=device)

    # Create loss function
    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT
    )

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        verbose=True,
        min_lr=1e-6
    )

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_iou = 0.0
    best_corner_error_percent = float('inf')
    history = {
        'train_loss': [],
        'train_iou': [],
        'train_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_corner_error_pixels': [],
        'val_corner_error_percent': [],
        'val_corner_success_rate': [],
        'lr': []
    }

    if resume_from is not None:
        checkpoint_path = None

        if resume_from == 'latest':
            # Find latest checkpoint
            checkpoints = list(config.CHECKPOINT_DIR.glob('checkpoint_epoch_*.pth'))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"Resuming from latest checkpoint: {checkpoint_path}")
            else:
                print("No checkpoints found, starting from scratch")

        elif resume_from == 'best':
            # Resume from best model
            if config.MODEL_PATH.exists():
                checkpoint_path = config.MODEL_PATH
                print(f"Resuming from best model: {checkpoint_path}")
            else:
                print("Best model not found, starting from scratch")

        else:
            # Resume from specific path
            checkpoint_path = Path(resume_from)
            if checkpoint_path.exists():
                print(f"Resuming from checkpoint: {checkpoint_path}")
            else:
                print(f"Checkpoint {checkpoint_path} not found, starting from scratch")

        # Load checkpoint if found
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded model from epoch {checkpoint['epoch']}")

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  Loaded optimizer state")

            # Set start epoch
            start_epoch = checkpoint['epoch'] + 1

            # Load best metrics if available
            if 'val_loss' in checkpoint:
                best_val_loss = checkpoint['val_loss']
                print(f"  Best val loss: {best_val_loss:.4f}")
            if 'val_iou' in checkpoint:
                best_val_iou = checkpoint['val_iou']
                print(f"  Best val IoU: {best_val_iou*100:.1f}%")
            if 'val_corner_error_percent' in checkpoint:
                best_corner_error_percent = checkpoint['val_corner_error_percent']
                print(f"  Best corner error: {best_corner_error_percent:.2f}%")

            # Try to load training history
            history_path = config.OUTPUT_DIR / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                print(f"  Loaded training history ({len(history['train_loss'])} epochs)")

            print(f"\nResuming training from epoch {start_epoch}")
        else:
            print("\nStarting training from scratch")

    if start_epoch == 1:
        print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    else:
        print(f"\nContinuing training from epoch {start_epoch} to {config.NUM_EPOCHS}...")

    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print()

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 60)

        # Train
        train_loss, train_iou, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        save_vis = (epoch % config.VAL_VIS_FREQUENCY == 0)
        val_results = validate_epoch(
            model, val_loader, criterion, device, config, epoch, save_vis
        )
        val_loss, val_iou, val_dice, val_corner_error_px, val_corner_error_pct, val_corner_success = val_results

        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['val_corner_error_pixels'].append(val_corner_error_px)
        history['val_corner_error_percent'].append(val_corner_error_pct)
        history['val_corner_success_rate'].append(val_corner_success)
        history['lr'].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou*100:.1f}%, Dice: {train_dice*100:.1f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou*100:.1f}%, Dice: {val_dice*100:.1f}%")
        print(f"  Val   - Corner Error: {val_corner_error_pct:.2f}% ({val_corner_error_px:.1f}px), Success: {val_corner_success:.1f}%")
        print(f"  LR: {current_lr:.6f}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_corner_error_pixels': val_corner_error_px,
                'val_corner_error_percent': val_corner_error_pct,
                'val_corner_success_rate': val_corner_success,
            }, config.MODEL_PATH)

        # Save best model based on validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            print(f"  ✓ New best validation IoU: {best_val_iou*100:.1f}%")

        # Track best corner error
        if val_corner_error_pct < best_corner_error_percent:
            best_corner_error_percent = val_corner_error_pct
            print(f"  ✓ New best corner error: {best_corner_error_percent:.2f}%")

        # Save checkpoint periodically
        if epoch % 50 == 0:
            checkpoint_path = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_corner_error_pixels': val_corner_error_px,
                'val_corner_error_percent': val_corner_error_pct,
                'val_corner_success_rate': val_corner_success,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Save training history
    history_path = config.OUTPUT_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation IoU: {best_val_iou*100:.1f}%")
    print(f"Best corner error: {best_corner_error_percent:.2f}%")
    print(f"Model saved to: {config.MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()
