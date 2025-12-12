"""
Command-line interface for paper segmentation training

Usage:
    python -m paper_detection.segmentation                    # Start from scratch
    python -m paper_detection.segmentation --resume latest    # Resume from latest checkpoint
    python -m paper_detection.segmentation --resume best      # Resume from best model
    python -m paper_detection.segmentation --resume <path>    # Resume from specific checkpoint
"""

import argparse
from paper_detection.segmentation.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train paper segmentation U-Net model')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint. Options: "latest", "best", or path to checkpoint file'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Paper Segmentation U-Net Training")
    print("=" * 60)

    train(resume_from=args.resume)
