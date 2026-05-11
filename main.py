#!/usr/bin/env python3
"""
Main entry point for image parser CLI

This is a convenience wrapper that calls parser/parse.py
"""

import sys
from pathlib import Path

# Add parser to path
current_dir = Path(__file__).parent
parser_dir = current_dir / "parser"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parser_dir))

if __name__ == "__main__":
    # Import and run parser CLI
    from parser.parse import main
    main()
