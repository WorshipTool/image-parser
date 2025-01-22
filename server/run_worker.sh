export QUEUE_NAME=parse_files
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
rq worker --with-scheduler $QUEUE_NAME