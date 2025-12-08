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
