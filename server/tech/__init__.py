"""
Queue management and file handling for image parser server.
"""

import os
import uuid

import redis
from rq import Queue

from ..constants import QUEUE_NAME, TEMP_FOLDER

# Connect to Redis
# Get Redis connection info from environment (for Docker support)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_conn = redis.Redis(host=redis_host, port=redis_port)

# Prepare Queue
q = Queue(QUEUE_NAME, connection=redis_conn)

# Upload folder configuration
UPLOAD_FOLDER = os.path.join(TEMP_FOLDER, "uploads")


from server.processor import processor


def add_to_queue(files: list[str], use_ai: bool):
    """
    Add a parsing job to the queue.

    Args:
        files: List of file paths to parse
        use_ai: Whether to use AI-based corrections

    Returns:
        RQ job object
    """
    job = q.enqueue(processor, files, use_ai)
    return job


def get_job(job_id: str):
    """
    Retrieve a job from the queue by ID.

    Args:
        job_id: Job identifier

    Returns:
        RQ job object or None
    """
    job = q.fetch_job(job_id)
    return job


def save_files(files: list):
    """
    Save received files to upload folder.

    Args:
        files: List of file objects to save

    Returns:
        List of saved file paths
    """
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    created_files = []
    for file in files:
        filename = os.path.join(UPLOAD_FOLDER, generate_filename(file.filename))
        file.save(filename)
        created_files.append(filename)

    return created_files


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
