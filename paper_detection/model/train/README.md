# Paper Corner Detection - Training

Basic PyTorch training implementation for corner detection model with Albumentations augmentation.

## Structure

```
model/train/
├── __init__.py          # Module init
├── config.py            # Training configuration
├── model.py             # Simple CNN model
├── dataset.py           # Dataset with Albumentations
├── train.py             # Training script
└── README.md            # This file
```

## Quick Start

```bash
# Train the model
python3 -m paper_detection.model.train
```

## Files

### config.py

Training configuration with dataclass:

-   Data paths (test images, ground truth JSON)
-   Training hyperparameters (batch size, epochs, learning rate)
-   Model parameters (image size, number of corners)
-   Device selection (CPU/CUDA)

### model.py

Simple CNN architecture:

-   4 convolutional blocks (32 → 64 → 128 → 256 channels)
-   Regression head (outputs 8 values: 4 corners × 2 coordinates)
-   BatchNorm and Dropout for regularization

### dataset.py

PyTorch Dataset with Albumentations:

-   Loads images and ground truth corners from JSON
-   Applies augmentation (brightness, contrast, rotation, flip, blur, noise)
-   Normalizes coordinates to 0-1 range
-   Uses keypoint transformations from Albumentations

### train.py

Basic training loop:

-   Train/validation split (80/20 by default)
-   MSE loss for corner coordinates
-   Adam optimizer
-   Saves best model based on validation loss
-   Reports pixel error metric

## Dependencies

```
torch
albumentations
opencv-python
numpy
tqdm
```

## Model Output

-   Input: RGB image [3, 224, 224]
-   Output: 8 normalized coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
-   Coordinates are in range [0, 1] relative to image size

## Training Output

Model checkpoint saved to:

```
paper_detection/model/checkpoints/best_model.pth
```

Checkpoint contains:

-   Model state dict
-   Optimizer state dict
-   Epoch number
-   Best validation loss
-   Best pixel error
