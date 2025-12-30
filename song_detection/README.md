# Song Detection Module

YOLOv8-based song detection for sheet music images.

## Features

- Detects 3 classes: `sheet`, `title`, `data` (chords/lyrics)
- Groups detections into logical song units
- Real-time detection support

## Setup

1. **Download model**:
   ```bash
   python song_detection/prepare.py
   ```
   This downloads `yolo8best.pt` (160MB) to project root.

2. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python
   ```

## Usage

### Python API

**Simple detection** (no progress tracking):
```python
import song_detection

# Prepare model
song_detection.prepare_model("yolo8best.pt")

# Detect songs in image (simple version)
results = song_detection.detect_simple("image.jpg", show=False)

# Process results
for song_group in results:
    if song_group.title:
        print(f"Title: {song_group.title.label}")
    if song_group.data:
        print(f"Data: {song_group.data.label}")
    if song_group.sheet:
        print(f"Sheet: {song_group.sheet.label}")
```

**Detection with progress tracking**:
```python
# detect() is a generator that yields progress (0-100)
detectGen = song_detection.detect("image.jpg", show=False)

while True:
    try:
        progress = next(detectGen)
        print(f"Progress: {progress}%")
    except StopIteration as e:
        results = e.value  # Final results
        break
```

### Command Line

**Show detections**:
```bash
python song_detection/detect_show.py image.jpg
```

**Real-time camera detection**:
```bash
python song_detection/realtime_detect.py
```

## Classes

### `SongDetectGroup`
Groups related detections:
- `title` - Song title detection
- `data` - Chords/lyrics detection
- `sheet` - Full sheet detection
- `image` - Cropped image of the group

### `CustomDetect`
Individual detection:
- `label` - Class name (sheet/title/data)
- `confidence` - Detection confidence [0-1]
- `bounds` - Bounding box (Bounds object)
- `image` - Cropped detection image

## Model Details

- **Architecture**: YOLOv8
- **Training**: Custom dataset of sheet music images
- **Classes**: 3 (sheet, title, data)
- **Input size**: 640x640 (auto-resized)
- **Pretrained from**: YOLOv8n (nano)

## Training

### Dataset Format

YOLO format with the following structure:

```
dataset/
├── dataset.yaml
├── train/
│   ├── images/
│   │   ├── photo1.jpg
│   │   └── ...
│   └── labels/
│       ├── photo1.txt      # class x_center y_center width height
│       └── ...
└── val/
    ├── images/
    └── labels/
```

**dataset.yaml**:
```yaml
path: /path/to/dataset
train: train/images
val: val/images

names:
  0: sheet
  1: title
  2: data
```

**Label format** (normalized 0-1):
```
0 0.5 0.5 0.8 0.9    # sheet: center_x, center_y, width, height
1 0.5 0.2 0.6 0.1    # title
2 0.5 0.6 0.7 0.7    # data
```

### Train Model

```bash
# Train from scratch (or continue from existing model)
./song_detection/train_detect.sh /path/to/dataset.yaml [optional_model.pt]

# This runs:
# yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml \
#      epochs=1000 imgsz=640 batch=8 patience=200
```

**Parameters**:
- `epochs=1000` - Maximum training epochs
- `imgsz=640` - Input image size
- `batch=8` - Batch size
- `patience=200` - Early stopping patience

### Model Compression (Optional)

Reduce model size using pruning:

```bash
python song_detection/prune.py input_model.pt output_model.pt

# Applies 80% sparsity to reduce model size
```

## Detection Pipeline

1. Load image
2. Run YOLO inference → detections with bounding boxes
3. Filter duplicates (remove smaller overlapping boxes)
4. Group detections:
   - Match title + data inside sheet
   - Match title + data by proximity
   - Keep standalone titles
5. Return list of `SongDetectGroup` objects

## Files

### Core Module
- `__init__.py` - Main module (prepare_model, detect, renderResults)
- `custom_detect.py` - CustomDetect class
- `song_detect_group.py` - SongDetectGroup and grouping logic

### CLI Tools
- `detect_show.py` - CLI tool to show detections on image
- `realtime_detect.py` - Real-time camera detection
- `prepare.py` - Model download script

### Training
- `train_detect.sh` - Training script
- `prune.py` - Model compression/pruning

### Tests
- `tests/test_detect.py` - Detection tests
- `tests/images/` - Test images
