import math
import os
import sys
from typing import Generator

import song_detection
import image_reader
import sheet_formatter
import common
from sheet_formatter.sheet import Sheet
import ai


# Prepare model
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "yolo8best.pt")
song_detection.prepare_model(model_path)

# Prepare paths
defaultOutputPath = os.path.join("tmp", "op.json")
defaultFormattedOutputPath = os.path.join("tmp", "fr.json")


def parse_images(inputImages: list[str], outputPath: str = None) -> Generator[int, None, list]:
    # AI analýza je vypnutá - bude se používat samostatně pro částečný processing
    useAi = False
    print("Basic Detecting")

    yield 0; # 0% progress

    ic = len(inputImages)

    

    # Loop over input images
    formattedResults = []
    for i, SAMPLE_IMAGE_PATH in enumerate(inputImages):

        partProgress = 0

        def getPartProgress():
            f = partProgress / ic + i * 100 / ic
            return math.floor(f)

        # Detect with generator stream process
        detectGen = song_detection.detect(SAMPLE_IMAGE_PATH, show=False)

        
        detectedResults = None

        # Handle generator stream
        while True:
            try:
                progress = next(detectGen)
                partProgress = progress * (3/10) # 0-30% progress
                yield getPartProgress()
            except StopIteration as e:
                # Zde se nachází finální návratová hodnota generátoru
                detectedResults = e.value
                break

        partProgress = 30 # 30% progress
        yield getPartProgress()

        if len(detectedResults) == 0:
            partProgress = 100 # 100% progress - no detected results
            yield getPartProgress()
            continue

        imageResults  = []
        for ii, detectedResult in enumerate(detectedResults):
            titleReadData = image_reader.read(detectedResult.title.image) if detectedResult.title is not None else None


            partProgress = 30 + ii * 30 / len(detectedResults) + 5 / len(detectedResults) # Cca 35% progress
            yield getPartProgress()

            dataReadData = image_reader.read(detectedResult.data.image) if detectedResult.data is not None else None



            partProgress = 30 + ii * 30 / len(detectedResults) + 10 / len(detectedResults) # Cca 40% progress
            yield getPartProgress()


            if(titleReadData is None or dataReadData is None):
                sheetReadData = image_reader.read(detectedResult.sheet.image) if detectedResult.sheet is not None else None
                if(sheetReadData is not None):
                    if(titleReadData is None):
                        titleReadData = sheetReadData
                    
                    if(dataReadData is None):
                        dataReadData = sheetReadData

                if(titleReadData is None and dataReadData is not None):
                    titleReadData = dataReadData

                
            if(titleReadData is None or dataReadData is None):
                continue

            partProgress = 30 + ii * 30 / len(detectedResults) + 15 / len(detectedResults) # Cca 45% progress
            yield getPartProgress()


            formatted = sheet_formatter.format(titleReadData, dataReadData, SAMPLE_IMAGE_PATH, detectedResult.image)

            print("\t" + str(len(imageResults)+1) + ". sheet detected")
            

            imageResults.append(formatted.to_json())

            partProgress = 30 + (ii+1) * 30 / len(detectedResults)  # 60% progress
            yield getPartProgress()



        partProgress = 60 # 60% progress
        yield getPartProgress()
        


        if(useAi):
            try:
                print("\nAnalyzing with AI...")
                common.write_json_to_file(imageResults, defaultFormattedOutputPath)
                response = ai.analyze_image(SAMPLE_IMAGE_PATH, defaultFormattedOutputPath)
                
                #add response items to formattedResults and .to_json() them
                for sheet in response:
                    formattedResults.append(sheet.to_json())
            except Exception as e:
                print("Error while working with AI...")
                
                # imageResult merge to formattedResults
                for imageResult in imageResults:
                    formattedResults.append(imageResult)

        else:
            # imageResult merge to formattedResults
            for imageResult in imageResults:
                formattedResults.append(imageResult)

        partProgress = 90 # 90% progress
        yield getPartProgress()

    yield 90 # 100% progress
    if outputPath is not None:
        # Write output to json file
        common.write_json_to_file(formattedResults, outputPath)

    yield 100 # 100% progress
    return formattedResults


# Use arguments when this script is called from command line
if __name__ == "__main__":
    # Arguments
    OUTPUT_JSON_PATH = ""
    INPUT_IMAGES_PATH = []

    # Load arguments
    allArgKeys = ["-o","-i"]

    OUTPUT_JSON_PATH = common.load_argument("-o", allArgKeys)
    INPUT_IMAGES_PATH = common.load_argument("-i", allArgKeys, True)
    gen = parse_images(INPUT_IMAGES_PATH, OUTPUT_JSON_PATH)

    # Handle generator stream
    while True:
        try:
            next(gen)  
        except StopIteration as e:
            break
    print("\nDone")
