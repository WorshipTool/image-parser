#!/usr/bin/env python3
"""
RQ Worker entry point for Docker

Starts an RQ worker to process background jobs from Redis queue.
"""

import os
import sys
from pathlib import Path

# Add paths
_current_dir = Path(__file__).parent
_image_parser_root = _current_dir.parent
sys.path.insert(0, str(_image_parser_root))

if __name__ == "__main__":
    from rq import Worker
    import redis
    import platform

    # Get Redis connection info from environment
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))

    # Connect to Redis
    redis_conn = redis.Redis(host=redis_host, port=redis_port)

    # Queue name
    queue_name = 'parse_files'

    print(f"Starting RQ worker...")
    print(f"  Redis: {redis_host}:{redis_port}")
    print(f"  Queue: {queue_name}")

    # Detect macOS and disable fork to avoid Objective-C runtime issues
    is_macos = platform.system() == 'Darwin'
    if is_macos:
        print("  Platform: macOS - using SimpleWorker (no fork)")
        os.environ['RQ_WORKER_CLASS'] = 'rq.worker.SimpleWorker'
        from rq.worker import SimpleWorker
        worker = SimpleWorker([queue_name], connection=redis_conn)
    else:
        print("  Platform: Linux/other - using standard Worker (with fork)")
        worker = Worker([queue_name], connection=redis_conn)

    # Start worker (RQ 2.x doesn't use Connection context manager)
    worker.work(with_scheduler=True)
