export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

QUEUE_NAME=parse_files
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $SCRIPT_DIR

cd ..

# Nastavení cesty k Pythonu a projektu
export PATH="$SCRIPT_DIR/../venv/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"

source venv/bin/activate


rq worker --with-scheduler $QUEUE_NAME