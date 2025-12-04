# Machine Learning Paper Detection Module

This module implements a deep learning-based paper corner detector using PyTorch. It can be used standalone or as an alternative/fallback to the traditional OpenCV-based detector.

## Overview

The ML module provides:
- **Lightweight CNN model** (CornerDetectorCNN) for direct corner regression
- **Training pipeline** with data augmentation for small datasets
- **Inference wrapper** (MLPaperDetector) compatible with the existing API
- **Integration** with PaperDetector class for seamless switching between methods

## Architecture

### Model: CornerDetectorCNN

A simple convolutional neural network designed for small datasets (~30 images):

```
Input: [B, 3, 224, 224] - RGB image, normalized with ImageNet stats
Conv Block 1: 3→32 channels
Conv Block 2: 32→64 channels
Conv Block 3: 64→128 channels
Conv Block 4: 128→256 channels
FC Layers: 256→128→8 (with dropout)
Output: [B, 8] - Normalized corner coordinates [x1,y1,x2,y2,x3,y3,x4,y4]
```

**Total parameters**: ~460K (intentionally small to avoid overfitting)

### Dataset: PaperCornersDataset

- Loads images and ground truth corners from JSON
- Applies geometric augmentations (flip, rotation) that transform corners accordingly
- Applies photometric augmentations (color jitter, blur)
- Normalizes with ImageNet statistics
- 80/20 train/val split by default

## Installation

The ML module requires PyTorch (already in requirements.txt):

```bash
pip install torch torchvision
```

Optional for training:
```bash
pip install tqdm tensorboard
```

## Usage

### 1. Training a Model

Train on the existing test dataset:

```bash
# From the image-parser directory
python -m paper_detection.ml.train

# With custom parameters
python -m paper_detection.ml.train \
    --epochs 300 \
    --batch-size 16 \
    --lr 0.001
```

**Training parameters** (see `config.py`):
- `image_size`: Input size (default: 224)
- `batch_size`: Batch size (default: 8)
- `num_epochs`: Training epochs (default: 200)
- `learning_rate`: Initial LR (default: 1e-3)
- `train_split`: Train/val split ratio (default: 0.8)
- `early_stopping_patience`: Early stopping patience (default: 30)

**Output**:
- Model checkpoint: `paper_detection/models/paper_detector_cnn.pth`
- Training history: `paper_detection/models/training_history.json`

### 2. Using Trained Model for Inference

#### Option A: Direct usage via MLPaperDetector

```python
from paper_detection.ml import MLPaperDetector
import cv2

# Initialize detector
detector = MLPaperDetector()

# Detect corners
image = cv2.imread("test.jpg")
corners = detector.predict_corners(image)

if corners is not None:
    print(f"Detected corners: {corners}")
    # corners shape: (4, 2) - [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

#### Option B: Via main PaperDetector class

```python
from paper_detection import PaperDetector
import cv2

# ML only
detector_ml = PaperDetector(detection_mode='ml')

# ML with OpenCV fallback (recommended)
detector_hybrid = PaperDetector(detection_mode='ml_with_fallback')

# Traditional OpenCV (default)
detector_cv = PaperDetector(detection_mode='opencv')

# Detect
image = cv2.imread("test.jpg")
corners = detector_hybrid.detect(image)
```

**Detection modes**:
- `"opencv"`: Traditional OpenCV method (default)
- `"ml"`: ML detection only
- `"ml_with_fallback"`: Try ML first, fallback to OpenCV if validation fails

### 3. Comparing Detection Methods

Compare OpenCV vs ML performance on test dataset:

```bash
# From paper_detection/tests directory
python compare_detection_methods.py

# With custom model
python compare_detection_methods.py --ml-model-path custom_model.pth
```

**Output**:
- `output/comparison/comparison_table.md`: Markdown comparison table
- `output/comparison/comparison_results.json`: Detailed results
- `output/comparison/comparison_statistics.json`: Summary statistics

## Training on Small Dataset (~30 images)

The current implementation is optimized for small datasets:

### Techniques Used

1. **Small Model Architecture**
   - Only ~460K parameters
   - Heavy dropout (0.5, 0.3)
   - Prevents overfitting

2. **Extensive Data Augmentation**
   - Horizontal flip (50%)
   - Vertical flip (50%)
   - Rotation ±15° (50%)
   - Color jitter (brightness, contrast, saturation, hue)
   - Gaussian blur

3. **Regularization**
   - Weight decay (0.01)
   - Dropout layers
   - Early stopping (patience=30)

4. **Validation-Based Training**
   - 80/20 train/val split
   - Learning rate scheduling (ReduceLROnPlateau)
   - Best model selection based on validation loss

### Expected Performance

With ~26 training images and ~7 validation images:
- Training will likely overfit after 50-100 epochs
- Early stopping will kick in around epoch 80-150
- Validation pixel error: 20-50px (varies by image difficulty)
- Some images may fail entirely due to limited training data

**⚠️ Important**: The ML model trained on ~30 images will likely:
- Work well on similar paper types and lighting conditions
- Struggle with significantly different scenarios
- Benefit greatly from more training data

## Improving Performance

### When You Have More Data (>100 images)

Update `model.py`:

```python
# Option 1: Use pretrained ResNet18 backbone
import torchvision.models as models

backbone = models.resnet18(pretrained=True)
# Replace final FC layer with corner regressor
```

### Adding More Training Data

1. Add images to `paper_detection/tests/test_images/`
2. Add ground truth to `test_corners_ground_truth.json`:
```json
{
  "new_image.jpg": {
    "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  // Normalized [0-1]
    "tolerance_pixels": 50
  }
}
```
3. Retrain: `python -m paper_detection.ml.train`

### Data Annotation Tool

Use the corner editor tool to annotate new images:

```bash
python paper_detection/tests/corner_editor.py
```

## File Structure

```
paper_detection/ml/
├── __init__.py              # Module exports
├── README.md                # This file
├── config.py                # Training/inference configuration
├── model.py                 # CornerDetectorCNN architecture
├── dataset.py               # PaperCornersDataset class
├── augmentations.py         # Geometric & photometric augmentations
├── train.py                 # Training script
└── inference.py             # MLPaperDetector wrapper

paper_detection/models/      # Model checkpoints (created after training)
└── paper_detector_cnn.pth   # Trained model weights
```

## API Reference

### MLPaperDetector

```python
class MLPaperDetector:
    def __init__(self, config: InferenceConfig = None)
    def predict_corners(self, image: np.ndarray) -> Optional[np.ndarray]
```

**Input**: BGR image (OpenCV format), any size
**Output**: (4, 2) array of corner coordinates or None

### PaperCornersDataset

```python
class PaperCornersDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        ground_truth_path: str,
        config: TrainingConfig,
        mode: str = "train",
        transform: callable = None
    )
```

**Returns**: Dictionary with keys:
- `"image"`: Tensor [3, 224, 224]
- `"corners"`: Tensor [8] - flattened coordinates
- `"image_name"`: str

### CornerDetectorCNN

```python
class CornerDetectorCNN(nn.Module):
    def __init__(self, dropout: float = 0.5)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Input**: [B, 3, 224, 224] - Normalized RGB
**Output**: [B, 8] - Normalized corner coordinates [0, 1]

## Troubleshooting

### Model file not found
```
FileNotFoundError: ML model not found at: paper_detection/models/paper_detector_cnn.pth
```
**Solution**: Train the model first using `python -m paper_detection.ml.train`

### Import error
```
ImportError: ML detection is not available
```
**Solution**: Install PyTorch: `pip install torch torchvision`

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: The inference wrapper automatically falls back to CPU. Or reduce batch size during training.

### Poor detection accuracy
- **Not enough training data**: Collect more images (aim for >100)
- **Different paper types**: Retrain with diverse examples
- **Extreme lighting**: Add more augmentation or training examples
- **Use hybrid mode**: `detection_mode='ml_with_fallback'` for best results

## TODO / Future Improvements

- [ ] Implement confidence score based on prediction uncertainty
- [ ] Add TensorBoard logging for training visualization
- [ ] Add perspective transform augmentation
- [ ] Implement K-fold cross-validation for better evaluation
- [ ] Use pretrained backbone (ResNet18) when more data available
- [ ] Add batch inference for multiple images
- [ ] Implement corner heatmap regression (instead of direct coordinates)
- [ ] Add model quantization for faster inference

## References

- Training script: `paper_detection/ml/train.py`
- Comparison tool: `paper_detection/tests/compare_detection_methods.py`
- Annotation tool: `paper_detection/tests/corner_editor.py`
- Main detector: `paper_detection/detector.py`
