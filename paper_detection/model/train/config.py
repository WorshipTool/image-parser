"""
Training configuration
"""

from dataclasses import dataclass
from pathlib import Path

from paper_detection.model.config import IMAGE_SIZE, NUM_CORNERS


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    # Data paths
    images_dir: Path = Path("paper_detection/data/images")
    corners_file: Path = Path("paper_detection/data/corners.json")

    # Output paths
    output_dir: Path = Path("paper_detection/model/checkpoints")

    # Training parameters
    batch_size: int = 2
    num_epochs: int = 200
    learning_rate: float = 0.001

    # Model parameters (from shared config)
    image_size: int = IMAGE_SIZE
    num_corners: int = NUM_CORNERS

    # Data split
    train_split: float = 0.8
    val_split: float = 0.2

    # Device
    device: str = "cpu"  # or "cuda" if available

    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
