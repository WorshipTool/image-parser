"""
Configuration for paper segmentation model
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for segmentation model training and inference"""

    # Model parameters
    IMAGE_SIZE: int = 384  # Input image size (384x384 is good balance for U-Net)
    NUM_CHANNELS: int = 3  # RGB input
    NUM_CLASSES: int = 1   # Binary segmentation (paper vs background)

    # U-Net architecture
    ENCODER_CHANNELS: list = None  # [64, 128, 256, 512]
    DECODER_CHANNELS: list = None  # [256, 128, 64, 32]

    # Training parameters
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 200
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5

    # Loss weights
    BCE_WEIGHT: float = 0.5
    DICE_WEIGHT: float = 0.5

    # Data paths
    IMAGES_DIR: Path = Path("paper_detection/data/images")
    CORNERS_FILE: Path = Path("paper_detection/data/corners.json")
    DATASET_FILE: Path = Path("paper_detection/data/dataset.json")

    # Output paths
    OUTPUT_DIR: Path = Path("paper_detection/models")
    CHECKPOINT_DIR: Path = Path("paper_detection/models/segmentation_checkpoints")
    DEBUG_DIR: Path = Path("temp/segmentation_debug")

    # Model save path
    MODEL_PATH: Path = Path("paper_detection/models/paper_segmentation_unet.pth")

    # Data split
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.2

    # Device
    DEVICE: str = "cpu"  # Will be set to "cuda" if available

    # Postprocessing parameters
    MASK_THRESHOLD: float = 0.5
    MIN_CONTOUR_AREA: int = 1000  # Minimum area for valid contour
    APPROX_EPSILON: float = 0.02  # Polygon approximation parameter

    # Augmentation parameters
    AUG_BRIGHTNESS_CONTRAST_P: float = 0.5
    AUG_HUE_SAT_P: float = 0.3
    AUG_BLUR_P: float = 0.3
    AUG_NOISE_P: float = 0.3
    AUG_ROTATE_P: float = 0.6
    AUG_HFLIP_P: float = 0.2
    AUG_VFLIP_P: float = 0.1
    AUG_PERSPECTIVE_P: float = 0.2

    # Validation visualization
    VAL_VIS_FREQUENCY: int = 10  # Save visualizations every N epochs
    NUM_VAL_VIS_SAMPLES: int = 4  # Number of validation samples to visualize

    def __post_init__(self):
        """Initialize derived parameters and create directories"""
        # Set default encoder/decoder channels if not provided
        if self.ENCODER_CHANNELS is None:
            self.ENCODER_CHANNELS = [64, 128, 256, 512]
        if self.DECODER_CHANNELS is None:
            self.DECODER_CHANNELS = [256, 128, 64, 32]

        # Create output directories
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
