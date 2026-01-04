
# Connect to Redis
import os
import redis

# Get Redis connection info from environment (for Docker support)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_conn = redis.Redis(host=redis_host, port=redis_port)

# Prepare Queue
from rq import Queue
from ..constants import QUEUE_NAME, TEMP_FOLDER
q = Queue(QUEUE_NAME,connection=redis_conn)


# Function to add a job
from server.processor import processor
def add_to_queue(files: list[str], useAi: bool):
    job = q.enqueue(processor, files, useAi)
    return job

def get_job(id: str):
    job = q.fetch_job(id)
    return job


# Function for save received files to folder
import os
UPLOAD_FOLDER = os.path.join(TEMP_FOLDER, "uploads")

def save_files(files: list):
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    createdFiles = []
    for file in files:
        filename = os.path.join(UPLOAD_FOLDER, generate_filename(file.filename))
        file.save(filename)
        createdFiles.append(filename)

    return createdFiles



# Generate random filename
import uuid
def generate_filename(filename):
    # Získání přípony souboru
    file_extension = os.path.splitext(filename)[1]
    
    # Generování náhodného názvu s uuid a přidání původní přípony
    random_filename = str(uuid.uuid4()) + file_extension
    
    return random_filename
