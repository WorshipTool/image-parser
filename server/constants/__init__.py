import os
from pathlib import Path

# Server constants
QUEUE_NAME = "parse_files"

# Temp folder path (relative to image-parser root)
_current_dir = Path(__file__).parent
_server_dir = _current_dir.parent
_image_parser_root = _server_dir.parent
TEMP_FOLDER = str(_image_parser_root / "temp")