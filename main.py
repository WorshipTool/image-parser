import os

import song_detection
import image_reader
import sheet_formatter
import common
from sheet_formatter.sheet import Sheet

# Arguments
OUTPUT_JSON_PATH = ""
INPUT_IMAGES_PATH = []

# Load arguments 
if(common.use_arguments()):
    allArgKeys = ["-o","-i"] 
    print("Using arguments")

    OUTPUT_JSON_PATH = common.load_argument("-o", allArgKeys)
    INPUT_IMAGES_PATH = common.load_argument("-i", allArgKeys, True)
else:
    print("Using default values")
print("\n")

# Prepare model
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "yolo8best.pt")
song_detection.prepare_model(model_path)
print("\n")

# Loop over input images
formattedResults = []
for SAMPLE_IMAGE_PATH in INPUT_IMAGES_PATH:
    detectedResults = song_detection.detect(SAMPLE_IMAGE_PATH, show=False)

    if len(detectedResults) == 0:
        continue


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

            

        formatted = sheet_formatter.format(titleReadData, dataReadData, SAMPLE_IMAGE_PATH, detectedResult.image)

        print(str(len(formattedResults)+1) + ". sheet detected")
        formattedResults.append(formatted.to_json())


# Write output to json file
common.write_json_to_file(formattedResults, OUTPUT_JSON_PATH)


print("\nDone")