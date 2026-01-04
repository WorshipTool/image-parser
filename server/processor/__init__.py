"""
Job processor for parsing image files.
"""

import os
import sys
from pathlib import Path
from typing import Generator

from rq import get_current_job

# Add parent paths
_current_dir = Path(__file__).parent
_server_dir = _current_dir.parent
_image_parser_root = _server_dir.parent
sys.path.insert(0, str(_image_parser_root))

from server.api import parse_images
from ..constants import TEMP_FOLDER

UPLOAD_FOLDER = os.path.join(TEMP_FOLDER, "uploads")


def parse_file_func(file_paths: list[str], use_ai: bool) -> Generator:
    """
    Process image files and parse their content.

    Args:
        file_paths: List of file paths to process
        use_ai: Whether to use AI-based corrections

    Yields:
        int: Progress percentage

    Returns:
        Parsed results or error message
    """
    # If no files provided, return error
    if len(file_paths) == 0:
        return {"message": "No files"}

    try:
        created_files = file_paths

        # Call parsing function using new parser API
        parse_gen = parse_images(created_files, use_ai=use_ai, debug=False)
        result = None

        # Handle generator stream
        while True:
            try:
                progress = next(parse_gen)
                yield progress
            except StopIteration as e:
                result = e.value
                break

        # Delete the uploaded files
        for file in created_files:
            os.remove(file)

        # inputImagePath is already basename in new parser API
        # No need to replace it

        return result
    except Exception as e:
        # Delete the uploaded files
        for file in created_files:
            if os.path.exists(file):
                os.remove(file)

        print(e)

        return {"message": str(e)}


def processor(files: list, use_ai: bool):
    """
    Redis queue processor for file parsing jobs.

    Args:
        files: List of file paths to process
        use_ai: Whether to use AI-based corrections

    Returns:
        Parsed results
    """
    job = get_current_job()
    gen = parse_file_func(files, use_ai)

    while True:
        try:
            progress = next(gen)
            job.meta['progress'] = progress
            job.save_meta()

        except StopIteration as e:
            return e.value
