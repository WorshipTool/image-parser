"""
Document Orientation Submodule

Text-based orientation detection for scanned documents using OCR.
Part of the paper_detection module.
"""

from .orient_text import orient_by_text, batch_orient_text

__all__ = ['orient_by_text', 'batch_orient_text']
