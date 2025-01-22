export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

QUEUE_NAME=parse_files
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $SCRIPT_DIR

cd ..

source venv/bin/activate

rq worker --with-scheduler $QUEUE_NAME