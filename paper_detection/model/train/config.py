"""
Training configuration
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    # Data paths
    images_dir: Path = Path("paper_detection/data/images")
    corners_file: Path = Path("paper_detection/data/corners.json")

    # Output paths
    output_dir: Path = Path("paper_detection/model/checkpoints")

    # Training parameters
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 0.001

    # Model parameters
    image_size: int = 224  # Input image size for model
    num_corners: int = 4   # 4 corners to detect

    # Data split
    train_split: float = 0.8
    val_split: float = 0.2

    # Device
    device: str = "cpu"  # or "cuda" if available

    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
