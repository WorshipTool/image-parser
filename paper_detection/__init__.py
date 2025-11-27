"""
Paper Detection Module

Isolated module for paper detection in images.
Detects paper using OpenCV and visualizes it with a blue frame.
"""

from .detector import PaperDetector
from .visualizer import PaperVisualizer

__all__ = ['PaperDetector', 'PaperVisualizer']
