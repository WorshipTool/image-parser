from PIL import Image
import cv2 as cv
import os

from .custom_detect import CustomDetect
from .photo_perspective_fixer import PhotoPerspectiveFixer
from ultralytics import YOLO 

modelReady = False
tempFolderPath = "tmp"

def prepare_model(modelPath: str):
    global model
    global modelReady
    model = YOLO(modelPath)
    modelReady = True

    # Create temp folder if not exists
    if not os.path.exists(tempFolderPath):
        os.makedirs(tempFolderPath)


def detect(imagePath: str,show: bool = False) -> list[CustomDetect]:
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
    results = model.predict(FIXED_INPUT_IMAGE_PATH)
    formattedResults : list[CustomDetect] = []
    for result in results:
        for box in result.boxes:
            formattedResults.append(CustomDetect(box, result))

    for result in formattedResults:
        if show:
            windowName = "image" + str(result.bounds.left) + str(result.bounds.top)
            cv.imshow(windowName, result.image)
            
            

    if show:
        cv.waitKey()

    #filter out non-sheet results
    formattedResults = list(filter(lambda x: x.label == "sheet", formattedResults))
    return formattedResults




def launchRealTimeDetection():
    if not modelReady:
        print("Model not ready. Please call prepare_model() first.")
        return

    cap = cv.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()


        # Detect
        results = model.predict(frame)
        # # Draw results
        for result in results:
            for box in result.boxes:
                cv.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
                cv.putText(frame, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        # Display the resulting frame
        cv.imshow('frame', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()