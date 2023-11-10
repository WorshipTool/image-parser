import torch
from PIL import Image
import cv2 as cv
import numpy as np

from .custom_detect import CustomDetect
from .photo_perspective_fixer import PhotoPerspectiveFixer

modelReady = False

def prepare_model(modelPath: str):
    global model
    global modelReady
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=modelPath, force_reload=False)
    modelReady = True

def detect(imagePath: str, tempFolderPath: str,show: bool = False) -> list[CustomDetect]:
    if not modelReady:
        print("Model not ready. Please call prepare_model() first.")
        return []

    # Fix rotation and perspective
    FIXED_INPUT_IMAGE_PATH = tempFolderPath + "/perspective-fixed.jpg"

    inputImage = cv.imread(imagePath)
    perspectiveFixedImage = PhotoPerspectiveFixer.fix(inputImage)
    cv.imwrite(FIXED_INPUT_IMAGE_PATH, perspectiveFixedImage)



    if show:
        Image.open(FIXED_INPUT_IMAGE_PATH).show()
        
    # Detect
    results = model(FIXED_INPUT_IMAGE_PATH)
    cropedImages = results.crop(save=False)


    formattedResults : list[CustomDetect] = []

    for crop in cropedImages:
        imageData = crop["im"]
        image = np.array(imageData, dtype=np.uint8)

        image = PhotoPerspectiveFixer.fix(image)

        if show:
            cv.imshow("image", image)
            cv.waitKey()

        formattedResults.append(CustomDetect(crop)) 

    
    return formattedResults

