

import os
import sys
from typing import Generator
from pathlib import Path

# Add parent paths
_current_dir = Path(__file__).parent
_server_dir = _current_dir.parent
_image_parser_root = _server_dir.parent
sys.path.insert(0, str(_image_parser_root))

from server.api import parse_images
from constants import TEMP_FOLDER

UPLOAD_FOLDER = os.path.join(TEMP_FOLDER, "uploads")


def parse_file_func(filePaths : list[str], useAi: bool) -> Generator[int, None, any]:

    # Pokud soubor nemá název, vrátíme chybu
    if len(filePaths) == 0:
        return {"message":"No files"}

    try:
        createdFiles = filePaths

        # Zavoláme funkci pro zpracování obrázku pomocí nového parser API
        parseGen = parse_images(createdFiles, use_ai=useAi, debug=False)
        result = None

        # Handle generator stream
        while True:
            try:
                progress = next(parseGen)
                yield progress
            except StopIteration as e:
                result = e.value
                break


        # Delete the uploaded files
        for file in createdFiles:
            os.remove(file)


        # inputImagePath is already basename in new parser API
        # No need to replace it

        return result
    except Exception as e:
        # Delete the uploaded files
        for file in createdFiles:
            if os.path.exists(file):
                os.remove(file)

        print(e)

        return {"message":str(e)}





# Function called from Redis
from rq import get_current_job
def processor(files: list, useAi: bool):
    job = get_current_job()
    gen = parse_file_func(files, useAi)

    while True:
        try:
            progress = next(gen)
            job.meta['progress'] = progress
            job.save_meta()


        except StopIteration as e:
            return e.value



