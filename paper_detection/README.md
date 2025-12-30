# Paper Detection

Neural network-based paper corner detection using U-Net segmentation.

## Quick Start

```python
from paper_detection import PaperDetector
import cv2

detector = PaperDetector()
image = cv2.imread("photo.jpg")
corners = detector.detect(image)

if corners is not None:
    print(f"Corners: {corners}")  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
```

## How It Works

**Approach:** Segmentation-based (not direct corner regression)

```
Image → U-Net → Paper Mask → Extract Corners → 4 Corners
```

1. Resize to 384×384
2. U-Net predicts paper probability mask
3. Threshold + morphological cleaning
4. Find contours → approximate to 4 corners
5. Order clockwise from top-left
6. Scale back to original resolution

## API

### `PaperDetector.detect(image, debug=False)`

Detect paper corners in image.

**Parameters:**
- `image` - BGR image (from cv2.imread)
- `debug` - Enable visualization (optional)

**Returns:**
- `np.ndarray` - Shape [4, 2], clockwise from top-left, pixel coordinates
- `None` - If paper not detected

## Model

**U-Net Binary Segmentation:**
- Input: RGB 384×384
- Architecture: 4-level encoder-decoder with skip connections
- Output: Binary mask (paper vs background)
- Weights: 160 MB (`model/checkpoints/paper_segmentation_unet.pth`)

**Training:**
- Dataset: 89 annotated photos in `data/images/`
- Ground truth: `data/corners.json` (normalized 0-1 coordinates)
- Loss: BCE + Dice (0.5 each)
- Augmentation: Color, geometric, perspective (Albumentations)

## Structure

```
paper_detection/
├── detector.py              # PaperDetector class
├── types.py                 # Type definitions
├── data/
│   ├── images/              # Training photos
│   └── corners.json         # Ground truth annotations
└── model/
    ├── model.py             # U-Net architecture
    ├── infer.py             # Inference engine
    ├── postprocess.py       # Mask → corners extraction
    ├── dataset.py           # Training dataset
    ├── config.py            # Configuration
    ├── checkpoints/
    │   └── paper_segmentation_unet.pth  # Trained model
    └── train/
        └── trainer.py       # Training loop
```

## Testing

```bash
pytest paper_detection/tests/ -v
```

## Integration

```python
from paper_detection import PaperDetector
from paper_transform import warp_paper, orient_by_text

# 1. Detect corners
detector = PaperDetector()
corners = detector.detect(photo)

# 2. Straighten
warped = warp_paper(photo, corners)

# 3. Orient
oriented, angle, conf = orient_by_text(warped)
```

## Performance

- Detection: ~0.2-0.5s per image (GPU much faster)
- Model size: 160 MB
- Input size: Any (resized to 384×384 internally)

## Why Segmentation?

✅ Robust to occlusion and extreme perspective
✅ Interpretable (visualize mask)
✅ Fallback strategies (convex hull, minAreaRect)
