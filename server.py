import time
from typing import Generator
from flask import Flask, Response, request, jsonify
import os
import uuid


# load envs
from dotenv import load_dotenv
load_dotenv()


PORT = os.getenv("PORT", 5000)
HOST = os.getenv("HOST", None)

# Connect to bridge
import bridge
bridge.start(PORT)


from main import parse_images

app = Flask(__name__)


# Enable CORS for all routes
from flask_cors import CORS
CORS(app)

current_directory = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(current_directory, "tmp/uploads")

# Set the maximum file size to 50MB
MEGABYTE = (2 ** 10) ** 2
app.config['MAX_CONTENT_LENGTH'] = 50 * MEGABYTE  
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * MEGABYTE

# Setup Swagger
from flasgger import Swagger, swag_from
swagger_config = {
    "specs_route": "/docs/",
    # "static_url_path":"/docs-json"
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/docs-json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
}
swagger = Swagger(app, swagger_config, merge=True)


# Queue dashboard 
import rq_dashboard
app.config.from_object("rq_dashboard.default_settings")
app.config["RQ_DASHBOARD_REDIS_URL"] = "redis://127.0.0.1:6379"
rq_dashboard.web.setup_rq_connection(app)
app.register_blueprint(rq_dashboard.blueprint, url_prefix="/board")



from server.tech import add_to_queue, save_files, get_job

@app.route('/is-available', methods=['GET'])
@swag_from("server/swagger/is-available.yml")
def is_available():
    return jsonify(isAvailable=True), 200



@app.route('/parse-file', methods=['POST'])
@swag_from("server/swagger/parse-file.yml")
def parse_file():

    useAi = request.args.get('useAi', default="false").lower() == "true"
    files = request.files.getlist('file')

    paths = save_files(files)


    job = add_to_queue(paths, useAi)
    

    # Wait until job is finished
    result = None
    while True:
        # Refresh job stav
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
@swag_from("server/swagger/parse-file.yml")
def parse_file_stream():
    
    useAi = request.args.get('useAi', default="false").lower() == "true"
    files = request.files.getlist('file')

    paths = save_files(files)

    job = add_to_queue(paths, useAi)

    return jsonify(id=job.id), 200

# Format func, job progress
def getProgressData(job):
    progress = job.meta.get('progress', 0)

    status = 4 # unknown

    if job.is_queued:
        status = 0 #"queued"
    elif job.is_started:
        status = 1 #"started"
    elif job.is_finished:
        status = 2 #"finished"
    elif job.is_failed:
        status = 3 #"failed"
    else:
        status = 4 #"unknown"

    data = {
        "progress": progress,
        "status": status
    }
    eventName = "progress"
    res = f"event: {eventName}\ndata: {data}"

    print("Progress", res)
    return res

@app.route("/get-job-status-stream", methods=['GET'])
@swag_from("server/swagger/get-job-status.yml")
def get_job_status_stream():
    job_id = request.args.get('id')
    job = get_job(job_id)

    if job is None:
        return jsonify(message="Job not found"), 404


    def stream():
        while True:
            job.refresh()
            yield getProgressData(job)
            if job.is_finished:
                yield f"event: final\ndata: {job.result}\n\n"
                break
            if job.is_failed:
                yield f"data: {job.exc_info}\n\n"
                break

            time.sleep(0.2)

    return Response(stream(), mimetype='text/event-stream')

@app.route("/get-job-result", methods=['GET'])
@swag_from("server/swagger/get-job-status.yml")
def get_job_result():
    job_id = request.args.get('id')
    job = get_job(job_id)

    if job is None:
        return jsonify(message="Job not found"), 404

    if job.is_finished:
        return jsonify(job.result), 200

    return jsonify(message="Job not finished yet"), 202



# Vytvoříme složku pro uploady, pokud neexistuje
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


app.run(debug=True, port=PORT, host=HOST)
