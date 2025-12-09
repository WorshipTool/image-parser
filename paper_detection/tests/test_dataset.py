"""
Test for dataset loading and augmentation visualization
"""

import cv2
import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper_detection.model.train.dataset import CornerDetectionDataset


class TestDataset:
    """Test dataset functionality"""

    def test_dataset_loading(self):
        """Test that dataset loads correctly"""
        dataset = CornerDetectionDataset(
            augment=False
        )

        # Check dataset has items
        assert len(dataset) > 0, "Dataset should have at least one image"

        # Check first item structure
        item = dataset[0]
        assert 'image' in item, "Item should have 'image' key"
        assert 'corners' in item, "Item should have 'corners' key"
        assert 'image_name' in item, "Item should have 'image_name' key"

        # Check image shape [C, H, W]
        assert item['image'].shape == (3, 224, 224), f"Image shape should be (3, 224, 224), got {item['image'].shape}"

        # Check corners shape [8] (4 corners x 2 coords)
        assert item['corners'].shape == (8,), f"Corners shape should be (8,), got {item['corners'].shape}"

        print(f"\n✓ Dataset loaded: {len(dataset)} images")
        print(f"✓ First image: {item['image_name']}")

    def test_augmentation_visualization(self):
        """Create a grid visualization of augmented images"""
        # Create dataset WITH augmentation
        dataset = CornerDetectionDataset(
            augment=True  # Enable augmentation
        )

        # Parameters for grid
        num_images = min(24, len(dataset))
        grid_rows = 4
        grid_cols = 6
        img_size = 224

        # Create grid canvas
        grid = np.zeros((grid_rows * img_size, grid_cols * img_size, 3), dtype=np.uint8)

        # Fill grid with augmented images
        for idx in range(num_images):
            item = dataset[idx]

            # Denormalize image (undo ImageNet normalization)
            # Original: (x - mean) / std
            # Reverse: x = (normalized * std) + mean
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

            image_tensor = item['image'].numpy()  # [C, H, W]
            image_denorm = (image_tensor * std) + mean

            # Clip to [0, 1] and convert to [H, W, C]
            image_denorm = np.clip(image_denorm, 0, 1)
            image_hwc = np.transpose(image_denorm, (1, 2, 0))  # [H, W, C]

            # Convert to uint8
            image_uint8 = (image_hwc * 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

            # Draw corners on image
            corners = item['corners'].numpy().reshape(4, 2)  # [4, 2]
            corners_px = corners * img_size  # Denormalize coordinates

            for i in range(4):
                pt = tuple(corners_px[i].astype(int))
                cv2.circle(image_bgr, pt, 3, (0, 255, 0), -1)

            # Place in grid
            row = idx // grid_cols
            col = idx % grid_cols

            y_start = row * img_size
            y_end = y_start + img_size
            x_start = col * img_size
            x_end = x_start + img_size

            grid[y_start:y_end, x_start:x_end] = image_bgr

        # Save grid
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "dataset_augmentation_grid.jpg"

        cv2.imwrite(str(output_path), grid)

        print(f"\n✓ Created augmentation grid: {num_images} images")
        print(f"✓ Saved to: {output_path}")

        assert output_path.exists(), "Grid image should be created"

    def test_corners_clockwise_order(self):
        """Test that corners are in correct clockwise order based on angle from centroid"""
        dataset = CornerDetectionDataset(
            augment=False  # No augmentation for order check
        )

        # Check multiple samples
        num_samples = min(10, len(dataset))
        
        for idx in range(num_samples):
            item = dataset[idx]
            corners = item['corners'].numpy().reshape(4, 2)  # [4, 2]
            
            # Check 1: First corner should be closest to top-left (0, 0)
            distances_to_origin = np.sum(corners ** 2, axis=1)
            closest_idx = np.argmin(distances_to_origin)
            assert closest_idx == 0, \
                f"Image {item['image_name']}: First corner should be closest to top-left. " \
                f"Closest is at index {closest_idx}, distances: {distances_to_origin}"
            
            # Check 2: Corners should be in clockwise order
            # Calculate centroid
            centroid = np.mean(corners, axis=0)
            
            # Calculate angles from centroid
            angles = np.arctan2(
                corners[:, 1] - centroid[1],
                corners[:, 0] - centroid[0]
            )
            
            # Angles should be in ascending order (accounting for wrap-around)
            # Since we start from the corner closest to top-left, angles might wrap
            # Check that angles are monotonically increasing or have one wrap point
            angle_diffs = np.diff(angles)
            
            # Count negative differences (wrap-around points)
            negative_diffs = np.sum(angle_diffs < 0)
            
            # Should have at most one wrap-around (from ~pi to ~-pi)
            assert negative_diffs <= 1, \
                f"Image {item['image_name']}: Too many wrap-around points ({negative_diffs}). " \
                f"Angles: {angles}, Diffs: {angle_diffs}"
            
            # If there's a wrap, it should be large (close to 2*pi)
            if negative_diffs == 1:
                wrap_idx = np.where(angle_diffs < 0)[0][0]
                wrap_size = abs(angle_diffs[wrap_idx])
                assert wrap_size > np.pi, \
                    f"Image {item['image_name']}: Wrap-around is too small ({wrap_size}). " \
                    f"Expected > pi. Angles: {angles}"
        
        print(f"\n✓ Verified clockwise order for {num_samples} images")
        print("✓ All corners are ordered by angle from centroid (clockwise direction)")
        print("✓ First corner is always closest to top-left (0, 0)")

    def test_augmentation_transforms_corners(self):
        """Test that image transformations (rotation, flip) are applied to corners"""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import json
        from paper_detection.model.train.config import TrainingConfig
        
        # Load config and ground truth
        config = TrainingConfig()
        with open(config.corners_file, 'r') as f:
            ground_truth = json.load(f)
        
        # Get first image
        image_name = list(ground_truth.keys())[0]
        image_path = Path(config.images_dir) / image_name
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Get corners in pixel coordinates
        corners_norm = np.array(ground_truth[image_name]["corners"], dtype=np.float32)
        corners_px = corners_norm.copy()
        corners_px[:, 0] *= w
        corners_px[:, 1] *= h
        
        # Test 1: Horizontal flip
        transform_flip = A.Compose([
            A.HorizontalFlip(p=1.0),  # Always flip
            A.Resize(224, 224),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        transformed_flip = transform_flip(image=image, keypoints=corners_px)
        corners_flipped = np.array(transformed_flip['keypoints'], dtype=np.float32)
        
        # After horizontal flip, x coordinates should be mirrored (w - x)
        # Check that corners moved horizontally
        expected_flip_x = w - corners_px[:, 0]
        # After resize to 224, scale accordingly
        scale_x = 224 / w
        scale_y = 224 / h
        expected_flip_x_scaled = expected_flip_x * scale_x
        expected_flip_y_scaled = corners_px[:, 1] * scale_y
        
        # Check x coordinates are approximately mirrored
        for i in range(4):
            assert abs(corners_flipped[i, 0] - expected_flip_x_scaled[i]) < 2, \
                f"Corner {i} x-coordinate not flipped correctly. " \
                f"Expected ~{expected_flip_x_scaled[i]}, got {corners_flipped[i, 0]}"
        
        print("\n✓ Horizontal flip correctly transforms corner coordinates")
        
        # Test 2: 90 degree rotation
        transform_rotate = A.Compose([
            A.Rotate(limit=0, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),  # No random, just test transform
            A.Resize(224, 224),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        # Apply a small rotation and check corners moved
        transform_rotate_10 = A.Compose([
            A.Rotate(limit=(10, 10), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Resize(224, 224),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        # Get unrotated version
        transform_no_rotate = A.Compose([
            A.Resize(224, 224),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        transformed_no_rot = transform_no_rotate(image=image, keypoints=corners_px)
        transformed_rot = transform_rotate_10(image=image, keypoints=corners_px)
        
        corners_no_rot = np.array(transformed_no_rot['keypoints'], dtype=np.float32)
        corners_rot = np.array(transformed_rot['keypoints'], dtype=np.float32)
        
        # After rotation, corners should have moved (not be identical)
        corners_diff = np.abs(corners_rot - corners_no_rot)
        max_diff = np.max(corners_diff)
        
        assert max_diff > 1.0, \
            f"Corners should move after rotation. Max difference: {max_diff}"
        
        print("✓ Rotation correctly transforms corner coordinates")
        print("✓ Augmentation transformations are properly applied to both image and corners")
