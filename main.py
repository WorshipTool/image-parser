import sys

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
song_detection.prepare_model("yolo8best.pt")
print("\n")

# Loop over input images
formattedResults = []
for SAMPLE_IMAGE_PATH in INPUT_IMAGES_PATH:
    detectedResults = song_detection.detect(SAMPLE_IMAGE_PATH, show=True)

    if len(detectedResults) == 0:
        continue


    for detectedResult in detectedResults:
        if detectedResult.label != "sheet":
            continue
        print(str(len(formattedResults)+1) + ". sheet detected")
        readData = image_reader.read(detectedResult.image)
        formatted = sheet_formatter.format(readData, SAMPLE_IMAGE_PATH, detectedResult.image)

        formattedResults.append(formatted.to_json())

# Write output to json file
common.write_json_to_file(formattedResults, OUTPUT_JSON_PATH)


print("\nDone")