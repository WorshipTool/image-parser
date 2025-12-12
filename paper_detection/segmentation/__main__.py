"""
Command-line interface for paper segmentation training

Usage:
    python -m paper_detection.segmentation
"""

from paper_detection.segmentation.train import train

if __name__ == "__main__":
    print("=" * 60)
    print("Paper Segmentation U-Net Training")
    print("=" * 60)
    train()
