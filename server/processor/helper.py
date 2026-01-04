"""
Helper utilities for processor module.
"""

import os
import uuid


def generate_filename(filename):
    """
    Generate random filename preserving extension.

    Args:
        filename: Original filename

    Returns:
        Random filename with original extension
    """
    # Extract file extension
    file_extension = os.path.splitext(filename)[1]

    # Generate random name with UUID and add original extension
    random_filename = str(uuid.uuid4()) + file_extension

    return random_filename
