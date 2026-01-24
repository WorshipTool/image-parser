"""
Flask server application for image parser.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, request, jsonify

# Load environment variables
load_dotenv()


PORT = os.getenv("PORT", 5000)
HOST = os.getenv("HOST", None)

# Add parent directory to path
_current_dir = Path(__file__).parent
_image_parser_root = _current_dir.parent
sys.path.insert(0, str(_image_parser_root))

app = Flask(__name__)

# Enable CORS for all routes
from flask_cors import CORS
CORS(app)

# Set paths relative to image-parser root
UPLOAD_FOLDER = os.path.join(str(_image_parser_root), "temp/uploads")

# Set the maximum file size to 50MB
MEGABYTE = (2 ** 10) ** 2
app.config['MAX_CONTENT_LENGTH'] = 50 * MEGABYTE
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * MEGABYTE

# Setup Swagger
from flasgger import Swagger, swag_from

swagger_config = {
    "specs_route": "/docs/",
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/docs-json',
            "rule_filter": lambda rule: True,  # All routes included
            "model_filter": lambda tag: True,  # All tags included
        }
    ],
}
swagger = Swagger(app, swagger_config, merge=True)

# Queue dashboard
import rq_dashboard
app.config.from_object("rq_dashboard.default_settings")
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
app.config["RQ_DASHBOARD_REDIS_URL"] = f"redis://{redis_host}:{redis_port}"
rq_dashboard.web.setup_rq_connection(app)
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/board")

from .tech import add_to_queue, save_files, get_job


@app.route('/is-available', methods=['GET'])
@swag_from("swagger/is-available.yml")
def is_available():
    """Check if service is available."""
    return jsonify(isAvailable=True), 200


@app.route('/parse-file', methods=['POST'])
@swag_from("swagger/parse-file.yml")
def parse_file():
    """Parse uploaded files synchronously."""
    use_ai = request.args.get('useAi', default="false").lower() == "true"
    files = request.files.getlist('file')

    paths = save_files(files)

    job = add_to_queue(paths, use_ai)

    # Wait until job is finished
    result = None
    while True:
        # Refresh job status
        job.refresh()

        if job.is_finished:
            result = job.result
            break
        if job.is_failed:
            err = job.exc_info
            return jsonify(message=err), 500

        time.sleep(0.5)

    return result, 200


@app.route('/add-file-to-parse-queue', methods=['POST'])
@swag_from("swagger/parse-file.yml")
def parse_file_stream():
    """Add files to parse queue and return job ID."""
    use_ai = request.args.get('useAi', default="false").lower() == "true"
    files = request.files.getlist('file')

    paths = save_files(files)

    job = add_to_queue(paths, use_ai)

    job.meta["useAi"] = use_ai
    job.save_meta()

    return jsonify(id=job.id), 200


def get_progress_data(job):
    """
    Format job progress data for streaming.

    Args:
        job: RQ job object

    Returns:
        Formatted progress string for SSE
    """
    progress = job.meta.get('progress', 0)

    status = 4  # Unknown

    if job.is_queued:
        status = 0  # Queued
    elif job.is_started:
        status = 1  # Started
    elif job.is_finished:
        status = 2  # Finished
    elif job.is_failed:
        status = 3  # Failed
    else:
        status = 4  # Unknown

    data = {
        "progress": progress,
        "status": status
    }
    eventName = "progress"
    res = f"event: {eventName}\ndata: {json.dumps(data)}\n\n"
    return res


@app.route("/get-job-status-stream", methods=['GET'])
@swag_from("swagger/get-job-status.yml")
def get_job_status_stream():
    """
    Stream job status updates via Server-Sent Events.

    Returns:
        Response with SSE stream
    """
    job_id = request.args.get('id')
    job = get_job(job_id)

    if job is None:
        return jsonify(message="Job not found"), 404

    def stream():
        while True:
            job.refresh()
            yield get_progress_data(job)
            if job.is_finished:

                use_ai = job.meta.get("useAi", False)

                data = {
                    "sheets": job.result,
                    "useAi": use_ai
                }

                yield f"event: final\ndata: {json.dumps(data)}\n\n"
                break
            if job.is_failed:
                yield f"event: error\ndata: {json.dumps({'error': str(job.exc_info)})}\n\n"
                break

            time.sleep(0.2)

    return Response(stream(), mimetype='text/event-stream')


@app.route("/get-job-result", methods=['GET'])
@swag_from("swagger/get-job-status.yml")
def get_job_result():
    """
    Get job result if finished.

    Returns:
        Job result with parsed sheets or status message
    """
    job_id = request.args.get('id')
    job = get_job(job_id)

    if job is None:
        return jsonify(message="Job not found"), 404

    use_ai = job.meta.get("useAi", False)

    if job.is_finished:
        return jsonify({
            "sheets": job.result,
            "useAi": use_ai
        }), 200

    return jsonify(message="Job not finished yet"), 202


# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
