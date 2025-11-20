#!/bin/bash
# Wrapper script pro snadné spuštění image preprocessing
# Použití: ./preprocess.sh -i obrazek.jpg [-o output.png]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Chyba: Virtual environment nebyl nalezen!"
    echo "   Spusť: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

exec "$VENV_PYTHON" -m image_preprocessing "$@"
