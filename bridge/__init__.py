import os
import requests
import schedule
import time
import threading


# load env
from dotenv import load_dotenv
load_dotenv()

bridgeUrl = os.getenv("BRIDGE_URL")
tickEndpointUrl = bridgeUrl + "/tick"
registerEndpointUrl = bridgeUrl + "/register"

bridgeServiceName = os.getenv("BRIDGE_SERVICE_NAME")
BRIDGE_SERVICE_TYPE = "parser"

id: str = None
port: int = None


def tick():
    try:

        global id
        if id is None:
            register()
        
        if id:
            response = requests.post(tickEndpointUrl, json={
                "id": id
            })

            if not response.ok:
                print("Failed to tick", response.text)
                id = None
                return
    except Exception as e:
        id = None
        print("Failed to tick", e)

def register():
    try:
        global port
        response = requests.post(registerEndpointUrl, json={
            "type": BRIDGE_SERVICE_TYPE,
            "name": bridgeServiceName,
            "connectVia": {
                "port": port
            }
        })

        if not response.ok:
            print("Failed to register to bridge", response.text)
            return

        data = response.json()

        global id
        id = data["id"]
    except Exception as e:
        print("Failed to register to bridge", e)


def call_api_periodically():

    tick()

    schedule.every(1).minutes.do(tick)
    while True:
        schedule.run_pending()
        time.sleep(1)

# Spuštění API volání na pozadí v novém vlákně
def start(listenPort: int):
    global port
    port = listenPort

    thread = threading.Thread(target=call_api_periodically, daemon=True)
    thread.daemon = True  # Ujistíme se, že tento thread bude ukončen při vypnutí programu
    thread.start()
