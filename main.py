import sys

import song_detection
import image_reader
import sheet_formatter
import common
from sheet_formatter.sheet import Sheet

# Arguments
TEMP_FOLDER_PATH = "/Users/pepavlin/Workspace/Programming/WorshipTool/data/temp"
CUSTOM_MODEL_PATH = "/Users/pepavlin/Workspace/Programming/WorshipTool/importers/old-image-parser/detection/models/240frames.pt"
OUTPUT_JSON_PATH = "/Users/pepavlin/Workspace/Programming/WorshipTool/data/temp/output.json"
INPUT_IMAGES_PATH = [
    "/Users/pepavlin/Workspace/Programming/WorshipTool/data/train_data/train_images/photos/IMG_20230826_093325.jpg",
    "/Users/pepavlin/Workspace/Programming/WorshipTool/data/train_data/train_images/photos/IMG_20230826_092914.jpg",
    "/Users/pepavlin/Workspace/Programming/WorshipTool/data/train_data/train_images/photos/IMG_20230826_093640.jpg",
    "/Users/pepavlin/Workspace/Programming/WorshipTool/data/train_data/train_images/photos/IMG_20230826_094216.jpg",
    "/Users/pepavlin/Workspace/Programming/WorshipTool/data/train_data/train_images/photos/IMG_20230826_093159.jpg",
    "/Users/pepavlin/Workspace/Programming/WorshipTool/data/train_data/train_images/scans/songbook/IMG_0001.jpg"
]

# Load arguments 
if(common.use_arguments()):
    allArgKeys = ["-m","-o","-t","-i"] 
    print("Using arguments")

    CUSTOM_MODEL_PATH = common.load_argument("-m", allArgKeys)
    OUTPUT_JSON_PATH = common.load_argument("-o", allArgKeys)
    TEMP_FOLDER_PATH = common.load_argument("-t", allArgKeys)
    INPUT_IMAGES_PATH = common.load_argument("-i", allArgKeys, True)
else:
    print("Using default values")
print("\n")

# Prepare model
song_detection.prepare_model(CUSTOM_MODEL_PATH)
print("\n")

# Loop over input images
formattedResults = []
for SAMPLE_IMAGE_PATH in INPUT_IMAGES_PATH:
    detectedResults = song_detection.detect(SAMPLE_IMAGE_PATH, TEMP_FOLDER_PATH, show=False)

    if len(detectedResults) == 0:
        continue


    for detectedResult in detectedResults:
        print(str(len(formattedResults)+1) + ". sheet detected")
        readData = image_reader.read(detectedResult.image)
        formatted = sheet_formatter.format(readData, SAMPLE_IMAGE_PATH, detectedResult.image)

        formattedResults.append(formatted.to_json())

# Write output to json file
common.write_json_to_file(formattedResults, OUTPUT_JSON_PATH)


print("\nDone")