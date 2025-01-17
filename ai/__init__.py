import requests
import base64
import json
import re

from sheet_formatter import Sheet

from openai import OpenAI
from dotenv import load_dotenv
import os

# Načtení proměnných z .env souboru
load_dotenv()

# Nastavení API klíče
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=api_key
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Funkce pro odeslání obrázku a dotazu na OpenAI API
def send_image_and_question(image_path, question):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def fix_json_input(malformed_json_string):
    # Odstranění přebytečných čárek před uzavíracími závorkami
    malformed_json_string = re.sub(r',\s*([\]}])', r'\1', malformed_json_string)
    return malformed_json_string

def parse_sheets_from_response(response, image_input_path) -> list[Sheet]:
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if not json_match:
        raise ValueError("JSON part not found in the provided text.")
    
    json_text = json_match.group(0)
    json_text = fix_json_input(json_text)
    
    # Load the JSON text into a Python object
    songs = json.loads(json_text)
    
    # Extract titles and data into the desired format
    parsed_songs = []
    for song in songs:
        parsed_song = Sheet(song.get("title"), song.get("data"), image_input_path, None)
        parsed_songs.append(parsed_song)
    
    return parsed_songs

def analyze_image(image_path, detected_json_path) -> list[Sheet]:
    question = "Analyzuj obrázek a najdi jednotlivé texty písně s akordy. Vrať výsledek jako JSON pole, kde každý prvek obsahuje objekt s názvem písně ('title') a obsahem('data'). Obsah má formát: sekce označ prvními písmeny názvů textu ve složených závorkách ({1S} - první sloka, {1R} - první refrén, {B} - bridge). Sekce se většinou liší jinými akordy, sekce má většinou více řádků. Všechny akordy, včetně složitějších nebo delších, vlož do hranatých závorek těsně před slovo, nad kterým jsou uvedeny, pokud ještě nejsou v textu zařazeny. Žádný akord nevynechej. Příklad: [Gm]Lala, [Amaj7][C#dim]tral. Zanech nové řádky, tam kde jsou i na obrázku."
    
    if detected_json_path is None:
        return parse_sheets_from_response(send_image_and_question(image_path, question), image_path)

    # get detected file content
    with open(detected_json_path, 'r') as file:
        detected_json = file.read()

        question = f"{question} Jako inspiraci použij data z poskytnutého výstupu mého algoritmu, ale oprav jeho nepřesnosti a zajisti správné umístění akordů: {detected_json}"

        return parse_sheets_from_response(send_image_and_question(image_path, question), image_path)


# # Příklad použití
# image_path = '/Users/pepavlin/Workspace/Programming/WorshipTool/data/images/songs/IMG_0423.jpeg'  # Cesta k obrázku
# detected_json_path = '/Users/pepavlin/Desktop/temp/output.json'

# response = analyze_image(image_path, detected_json_path )

# # print
# print(response)