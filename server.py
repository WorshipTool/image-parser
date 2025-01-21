import time
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



def generate_filename(filename):
    # Získání přípony souboru
    file_extension = os.path.splitext(filename)[1]
    
    # Generování náhodného názvu s uuid a přidání původní přípony
    random_filename = str(uuid.uuid4()) + file_extension
    
    return random_filename

@app.route('/is-available', methods=['GET'])
@swag_from("server/swagger/is-available.yml")
def is_available():
    return jsonify(isAvailable=True), 200


@app.route('/parse-file', methods=['POST'])
@swag_from("server/swagger/parse-file.yml")
def upload_file():
    
    useAi = request.args.get('useAi', default="false").lower() == "true"


    files = request.files.getlist('file')
    # Pokud soubor nemá název, vrátíme chybu
    if len(files) == 0:
        return jsonify(message="No selected file"), 400
    
    try:
    
        # Uložíme soubor do složky
        createdFiles = []
        for file in files:
            filename = os.path.join(UPLOAD_FOLDER, generate_filename(file.filename))
            file.save(filename)
            createdFiles.append(filename)



        # Zavoláme funkci pro zpracování obrázku
        parseGen = parse_images(createdFiles, useAi=useAi)
        result = None

        # Handle generator stream
        while True:
            try:
                progress = next(parseGen)
                print("Průběh parsu:", progress)

                # yield progress
            except StopIteration as e:
                result = e.value
                break


        # Delete the uploaded files
        for file in createdFiles:
            os.remove(file)


        # Replace inputImagePath in result items with the original filename
        for item in result:
            # Find original filename, index in createdFiles is the same as the original in files
            index = createdFiles.index(item["inputImagePath"])
            originalFile = files[index]
            item["inputImagePath"] = originalFile.filename



        return jsonify(result), 200
    except Exception as e:

        print("AAAAA")
        # Delete the uploaded files
        for file in createdFiles:
            os.remove(file)

        print(e)
        print("BBBBB")

        return jsonify(message=str(e)), 500

# Vytvoříme složku pro uploady, pokud neexistuje
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/testStream', methods=['GET'])
def textStream():
    def generate():
        for i in range(100):
            # wait for a while
            time.sleep(0.1)
            yield str(i)
        yield "DONE"
    return Response(generate(), mimetype='text/event-stream')




app.run(debug=True, port=PORT, host=HOST)
