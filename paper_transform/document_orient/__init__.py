"""
Document Orientation Submodule

Text-based orientation detection for scanned documents using OCR.
Part of the paper_transform module.
"""

from .orient_by_text import orient_by_text, batch_orient_text

__all__ = ['orient_by_text', 'batch_orient_text']
