from flask import Flask, request, jsonify
from flasgger import Swagger
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

UPLOAD_FOLDER = os.path.join(current_directory, "uploads")
app.config['TEMP_FOLDER'] = UPLOAD_FOLDER

# Set the maximum file size to 50MB
MEGABYTE = (2 ** 10) ** 2
app.config['MAX_CONTENT_LENGTH'] = 50 * MEGABYTE  
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * MEGABYTE

# Setup Swagger
swagger_config = {
    "specs_route": "/docs/",
    "static_url_path":"/docs-json"
}
swagger = Swagger(app, swagger_config, merge=True)



def generate_filename(filename):
    # Získání přípony souboru
    file_extension = os.path.splitext(filename)[1]
    
    # Generování náhodného názvu s uuid a přidání původní přípony
    random_filename = str(uuid.uuid4()) + file_extension
    
    return random_filename

@app.route('/parse-file', methods=['POST'])
def upload_file():
    """
    Parse song from file (image)
    This endpoint allows you to upload an image file and parse it to get the song data.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        description: Soubor, který bude nahrán
        required: true
    responses:
      200:
        description: Soubor byl úspěšně nahrán
        content:
          application/json:
            schema:
              type: object
              properties:
                filename:
                  type: string
                  description: Název nahraného souboru
    """

    files = request.files
    # Zkontrolujeme, zda soubor je součástí žádosti
    if 'file' not in files:
        return jsonify(message="No file part"), 400

    
    # Pokud soubor nemá název, vrátíme chybu
    if len(files) == 0:
        return jsonify(message="No selected file"), 400
    
    # Uložíme soubor do složky
    createdFiles = []
    for file in files.getlist('file'):
        filename = os.path.join(app.config['TEMP_FOLDER'], generate_filename(file.filename))
        file.save(filename)
        createdFiles.append(filename)

    # Zavoláme funkci pro zpracování obrázku
    result = parse_images(createdFiles, useAi=False)

    # Delete the uploaded files
    for file in createdFiles:
        os.remove(file)

    # Replace inputImagePath in result items with the original filename
    for item in result:
        # Find original filename, index in createdFiles is the same as the original in files
        index = createdFiles.index(item["inputImagePath"])
        originalFile = files.getlist('file')[index]
        item["inputImagePath"] = originalFile.filename



    return jsonify(result), 200

@app.route('/is-available', methods=['GET'])
def is_available():
    """
    Check if the server is available
    This endpoint allows you to check if the server is available.
    ---
    responses:
      200:
        description: Server is available
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: boolean
                  description: Server is available
    """
    return jsonify(isAvailable=True), 200

# Vytvoříme složku pro uploady, pokud neexistuje
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


app.run(debug=True, port=PORT, host=HOST)
