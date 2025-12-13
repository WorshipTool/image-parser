# Paper Segmentation Module

U-Net based model for robust paper detection with corner extraction.

## Overview

This module implements a **model-based** approach to paper corner detection, replacing the previous regression-based method. The pipeline consists of:

1. **U-Net Model**: Predicts a binary mask of the paper region
2. **Post-processing**: Extracts 4 corners from the mask using OpenCV contour analysis
3. **Integration**: Seamlessly integrates with existing `paper_transform` module

### Key Features

- ✅ More robust than direct corner regression
- ✅ Handles partial occlusion and difficult perspectives better
- ✅ Automatic mask generation from existing corner annotations
- ✅ Comprehensive augmentation pipeline
- ✅ IoU and Dice metrics tracking
- ✅ Learning rate scheduling
- ✅ Visualization of predictions during training
- ✅ Deterministic and debuggable

## Architecture

### U-Net Model

```
Input: RGB image [3, 384, 384]
  ↓
Encoder: 4 downsampling blocks (64 → 128 → 256 → 512)
  ↓
Bottleneck: 512 channels
  ↓
Decoder: 4 upsampling blocks with skip connections (512 → 256 → 128 → 64)
  ↓
Output: Binary mask logits [1, 384, 384]
```

**Loss Function**: Combined BCE + Dice Loss
- BCE: Good for pixel-wise classification
- Dice: Good for handling class imbalance

### Post-processing Pipeline

```
Predicted Mask (probability)
  ↓
Threshold (0.5)
  ↓
Morphological cleaning (close + open)
  ↓
Find contours
  ↓
Select largest contour
  ↓
Approximate polygon (Douglas-Peucker)
  ↓
Extract 4 corners
  ↓
Order clockwise (starting from top-left)
  ↓
Scale to original image resolution
```

## Installation

Required dependencies (add to `requirements.txt`):

```txt
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
albumentations>=1.3.0
numpy>=1.24.0
tqdm>=4.65.0
```

Install:
```bash
pip install -r requirements.txt
```

## Dataset Format

The module reuses existing corner annotations:

### Directory Structure
```
paper_detection/data/
├── images/                    # Training images
│   ├── IMG_001.jpg
│   ├── IMG_002.jpg
│   └── ...
├── corners.json              # Corner annotations (normalized 0-1)
└── dataset.json             # Optional: list of image names to use
```

### Ground Truth Format

**corners.json**:
```json
{
  "IMG_001.jpg": {
    "corners": [
      [0.62, 0.184],   // [x, y] normalized to [0, 1]
      [0.883, 0.536],
      [0.224, 0.853],
      [-0.04, 0.49]
    ]
  },
  ...
}
```

Ground truth masks are **automatically generated** from corners using `cv2.fillPoly()`.

## Training

### Quick Start

```bash
# Train with default settings
python -m paper_detection.model
```

### Configuration

Edit `paper_detection/model/config.py`:

```python
@dataclass
class SegmentationConfig:
    # Model
    IMAGE_SIZE: int = 384           # Input image size

    # Training
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 200
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5

    # Loss weights
    BCE_WEIGHT: float = 0.5
    DICE_WEIGHT: float = 0.5

    # Post-processing
    MASK_THRESHOLD: float = 0.5
    MIN_CONTOUR_AREA: int = 1000
    APPROX_EPSILON: float = 0.02

    # Data paths
    IMAGES_DIR: Path = Path("paper_detection/data/images")
    CORNERS_FILE: Path = Path("paper_detection/data/corners.json")
    MODEL_PATH: Path = Path("paper_detection/models/paper_model_unet.pth")
```

### Training Output

The training script produces:

1. **Model checkpoint**: `paper_detection/models/paper_model_unet.pth`
2. **Training history**: `paper_detection/models/training_history.json`
3. **Visualizations**: `temp/model_debug/val_predictions/`
4. **Checkpoints**: `paper_detection/models/model_checkpoints/` (every 50 epochs)

### Metrics Tracked

- **Loss**: Combined BCE + Dice
- **IoU** (Intersection over Union): Measures mask overlap
- **Dice Coefficient**: Alternative overlap metric
- **Learning Rate**: Automatically adjusted by ReduceLROnPlateau scheduler

### Visualization

Validation predictions are saved every 10 epochs to `temp/model_debug/val_predictions/`:

```
Original | Ground Truth Mask | Predicted Mask + Corners
```

## Inference

### Using PaperDetector API

```python
from paper_detection import PaperDetector
import cv2

# Create detector
detector = PaperDetector()

# Load image
image = cv2.imread("test_image.jpg")

# Detect corners
corners = detector.detect(image)  # Returns [4, 2] array or None

if corners is not None:
    print(f"Detected corners: {corners}")
    # corners are in original image coordinates
```

### Direct Segmentation Inference

```python
from paper_detection.model.infer import SegmentationInference
import cv2

# Create inference engine
inference = SegmentationInference()

# Load image
image = cv2.imread("test_image.jpg")

# Detect corners
corners = inference.detect_corners(image, debug=True)

# Or get mask + corners
mask_prob, mask_resized = inference.predict_mask(image)
```

### Visualization

```python
from paper_detection.model.infer import SegmentationInference
from pathlib import Path

inference = SegmentationInference()
image = cv2.imread("test_image.jpg")

# Detect and visualize
corners = inference.detect_corners(image)
mask_prob, _ = inference.predict_mask(image)

vis = inference.visualize_detection(
    image, corners, mask_prob,
    output_path=Path("temp/detection_result.jpg")
)
```

## Integration with paper_transform

The model module is **fully compatible** with the existing `paper_transform` module:

```python
from paper_detection import PaperDetector
from paper_transform import warp_document, orient_document
import cv2

# Detect corners
detector = PaperDetector()
image = cv2.imread("photo.jpg")
corners = detector.detect(image)

if corners is not None:
    # Warp using existing pipeline
    warped = warp_document(image, corners)

    # Orient using existing pipeline
    oriented = orient_document(warped)
```

## Testing

### Run All Tests

```bash
# Run all model tests
pytest paper_detection/model/tests/ -v

# Run specific test file
pytest paper_detection/model/tests/test_postprocess.py -v

# Run with coverage
pytest paper_detection/model/tests/ --cov=paper_detection.model
```

### Unit Tests

- `test_postprocess.py`: Tests for corner extraction and utilities
- `test_model.py`: Tests for U-Net architecture and losses
- `test_integration.py`: End-to-end tests

### Integration Tests

Integration tests require:
- Trained model at `paper_detection/models/paper_model_unet.pth`
- Test images in `paper_detection/data/images/`
- Ground truth in `paper_detection/data/corners.json`

```bash
pytest paper_detection/model/tests/test_integration.py -v
```

## Module Structure

```
paper_detection/model/
├── __init__.py              # Public API exports
├── __main__.py             # CLI training entry point
├── config.py               # Configuration dataclass
├── model.py                # U-Net architecture + losses
├── dataset.py              # Dataset with mask generation
├── postprocess.py          # Mask → corners extraction
├── train.py                # Training loop
├── infer.py                # Inference engine
├── README.md               # This file
└── tests/
    ├── __init__.py
    ├── test_postprocess.py    # Unit tests for post-processing
    ├── test_model.py          # Unit tests for model
    └── test_integration.py    # Integration tests
```

## Performance Tips

### Training

1. **GPU Training**: Automatically uses CUDA if available
   ```python
   # Check GPU availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

2. **Batch Size**: Increase if you have more GPU memory
   ```python
   config.BATCH_SIZE = 16  # Default: 8
   ```

3. **Image Size**: Larger = better quality but slower
   ```python
   config.IMAGE_SIZE = 512  # Default: 384
   ```

4. **Early Stopping**: Monitor validation loss and stop when plateaus

### Inference

1. **Batch Processing**: Process multiple images
   ```python
   inference = SegmentationInference()
   results = [inference.detect_corners(img) for img in images]
   ```

2. **Caching**: Reuse inference engine
   ```python
   # Good: Create once
   detector = PaperDetector()
   for image in images:
       corners = detector.detect(image)

   # Bad: Create multiple times
   for image in images:
       detector = PaperDetector()  # Slow!
       corners = detector.detect(image)
   ```

## Troubleshooting

### Issue: Model not found

```
FileNotFoundError: Model not found: paper_detection/models/paper_model_unet.pth
```

**Solution**: Train the model first:
```bash
python -m paper_detection.model
```

### Issue: No corners detected

```python
corners = detector.detect(image)  # Returns None
```

**Solutions**:
1. Enable debug mode: `detector.detect(image, debug=True)`
2. Adjust post-processing parameters in `config.py`:
   ```python
   config.MASK_THRESHOLD = 0.3  # Lower threshold
   config.MIN_CONTOUR_AREA = 500  # Lower minimum area
   config.APPROX_EPSILON = 0.03  # Higher epsilon for simpler polygon
   ```

### Issue: Poor accuracy

**Solutions**:
1. Train for more epochs
2. Increase image size: `config.IMAGE_SIZE = 512`
3. Add more training data
4. Adjust augmentation parameters
5. Check ground truth quality

### Issue: Training too slow

**Solutions**:
1. Reduce image size: `config.IMAGE_SIZE = 256`
2. Reduce batch size if GPU OOM: `config.BATCH_SIZE = 4`
3. Use GPU if available
4. Reduce number of epochs for testing

## Corner Ordering

Corners are **always** ordered:
1. **Clockwise** direction
2. Starting from corner **closest to top-left** (0, 0)

Example:
```
(0,0) ─────────────────→ x
  │
  │    TL ●────────● TR
  │       │        │
  │       │ Paper  │
  │       │        │
  │    BL ●────────● BR
  ↓
  y

Corners: [TL, TR, BR, BL]
```

This ensures consistent ordering across all images.

## Comparison: Segmentation vs Regression

| Aspect | Segmentation (New) | Regression (Old) |
|--------|-------------------|------------------|
| **Robustness** | ✅ More robust | ❌ Less robust |
| **Partial occlusion** | ✅ Handles well | ❌ Struggles |
| **Extreme perspective** | ✅ Better | ❌ Worse |
| **Training data** | ✅ Reuses corners | ✅ Reuses corners |
| **Inference speed** | ⚠️ Slightly slower | ✅ Faster |
| **Model size** | ⚠️ Larger (~50MB) | ✅ Smaller (~10MB) |
| **Interpretability** | ✅ Visual mask | ❌ Black box |

**Recommendation**: Use model mode for production.

## API Compatibility

The model module maintains **full backward compatibility** with existing code:

```python
from paper_detection import PaperDetector
detector = PaperDetector()
corners = detector.detect(image)
```

## Contributing

When adding features:
1. Add unit tests in `tests/`
2. Update this README
3. Ensure backward compatibility
4. Run full test suite: `pytest paper_detection/model/tests/`

## License

Same as parent project.
