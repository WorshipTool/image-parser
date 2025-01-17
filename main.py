import os
import sys

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


def parse_images(inputImages: list[str], outputPath: str = defaultOutputPath, useAi: bool = False): 
    if(useAi):
        print("Detecting with AI (Good 👍 )")
    else:
        print("Basic Detecting (Bad 😞 )")

    

    # Loop over input images
    formattedResults = []
    for SAMPLE_IMAGE_PATH in inputImages:
        detectedResults = song_detection.detect(SAMPLE_IMAGE_PATH, show=False)


        if len(detectedResults) == 0:
            continue

        imageResults  = []
        for detectedResult in detectedResults:
            titleReadData = image_reader.read(detectedResult.title.image) if detectedResult.title is not None else None
            dataReadData = image_reader.read(detectedResult.data.image) if detectedResult.data is not None else None

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

            formatted = sheet_formatter.format(titleReadData, dataReadData, SAMPLE_IMAGE_PATH, detectedResult.image)

            print("\t" + str(len(imageResults)+1) + ". sheet detected")
            

            imageResults.append(formatted.to_json())
        

        common.write_json_to_file(imageResults, defaultFormattedOutputPath)

        if(useAi):
            print("\nAnalyzing with AI...")
            response = ai.analyze_image(SAMPLE_IMAGE_PATH, defaultFormattedOutputPath)
            
            #add response items to formattedResults and .to_json() them
            for sheet in response:
                formattedResults.append(sheet.to_json())
            
        else:
            # imageResult merge to formattedResults
            for imageResult in imageResults:
                formattedResults.append(imageResult)


    # Write output to json file
    common.write_json_to_file(formattedResults, outputPath)
    return formattedResults


# Use arguments when this script is called from command line
if __name__ == "__main__":
    # Arguments
    OUTPUT_JSON_PATH = ""
    INPUT_IMAGES_PATH = []

    # check if  arg contains '-ai'
    USE_AI = '-ai' in sys.argv

    # Load arguments 
    allArgKeys = ["-o","-i"] 

    OUTPUT_JSON_PATH = common.load_argument("-o", allArgKeys)
    INPUT_IMAGES_PATH = common.load_argument("-i", allArgKeys, True)
    parse_images(INPUT_IMAGES_PATH, OUTPUT_JSON_PATH, USE_AI)

    print("\nDone")