from flask import Flask, request, jsonify
from flasgger import Swagger
import os
import uuid

from main import parse_images

app = Flask(__name__)
swagger = Swagger(app)

UPLOAD_FOLDER = 'tmp/uploads'  # Složka pro uložení souborů
app.config['TEMP_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maximální velikost souboru: 16 MB


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
    # Zkontrolujeme, zda soubor je součástí žádosti
    if 'file' not in request.files:
        return jsonify(message="No file part"), 400

    file = request.files['file']
    
    # Pokud soubor nemá název, vrátíme chybu
    if file.filename == '':
        return jsonify(message="No selected file"), 400
    
    if not file:
        return jsonify(message="No file part"), 400
    
    # Uložíme soubor do složky
    filename = os.path.join(app.config['TEMP_FOLDER'], generate_filename(file.filename))
    file.save(filename)

    # Zavoláme funkci pro zpracování obrázku
    result = parse_images([filename], useAi=True)

    # Delete the uploaded file
    os.remove(filename)
    

    return jsonify(result), 200



# Vytvoříme složku pro uploady, pokud neexistuje
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.run(debug=True, port=5000)