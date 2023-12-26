import requests
import os

MODEL_DROPBOX_URL = 'https://www.dropbox.com/scl/fi/7ch7pyk7u2c40xnyvytok/best.pt?rlkey=1g7jp9z4pengj49q2cv39ghm5&dl=1'

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "yolo8best.pt")

def download_file(url, local_name):
    response = requests.get(url)
    with open(local_name, 'wb') as file:
        file.write(response.content)

if __name__ == "__main__":
    lokalni_nazev = model_path
    download_file(MODEL_DROPBOX_URL, lokalni_nazev)
    print('Model has been successfuly downloaded.')