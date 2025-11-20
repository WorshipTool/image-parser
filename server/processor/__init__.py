

import os
from typing import Generator

from main import parse_images
from constants import TEMP_FOLDER

UPLOAD_FOLDER = os.path.join(TEMP_FOLDER, "uploads")


def parse_file_func(filePaths : list[str], useAi: bool) -> Generator[int, None, any]:
    
    # Pokud soubor nemá název, vrátíme chybu
    if len(filePaths) == 0:
        return {"message":"No files"}
    
    try:
        createdFiles = filePaths



        # Zavoláme funkci pro zpracování obrázku
        parseGen = parse_images(createdFiles, useAi=useAi)
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


        # Replace inputImagePath
        for item in result:
            # Get basename from the path
            pathname = os.path.basename(item["inputImagePath"])
            item["inputImagePath"] = pathname



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



