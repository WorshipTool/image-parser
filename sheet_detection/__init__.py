from typing import Generator
from PIL import Image
import cv2 as cv
import os

from ultralytics import YOLO

from .song_detect_group import SongDetectGroup, groupCustomDetect
from .custom_detect import CustomDetect


modelReady = False
model = None

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
tempFolderPath = os.path.join(parent_directory, "tmp")

# Default model path
DEFAULT_MODEL_PATH = os.path.join(parent_directory, "yolo8best.pt")

def prepare_model(modelPath: str = None):
    """
    Prepare the YOLO model for detection.

    Args:
        modelPath: Optional path to model file. If not provided, uses default path.
    """
    global model
    global modelReady

    # Use default path if not provided
    if modelPath is None:
        modelPath = DEFAULT_MODEL_PATH

    if not os.path.exists(modelPath):
        print(f"Model not found at: {modelPath}")
        print("Please run prepare.py to download the model.")
        return

    model = YOLO(modelPath)
    modelReady = True

    # Create temp folder if not exists
    if not os.path.exists(tempFolderPath):
        os.makedirs(tempFolderPath)

# Auto-initialize model on import if it exists
if os.path.exists(DEFAULT_MODEL_PATH):
    prepare_model()


def detect_simple(imagePath: str, show: bool = False) -> list[SongDetectGroup]:
    """
    Simple detection without progress tracking.

    Args:
        imagePath: Path to image file
        show: If True, display results in window

    Returns:
        List of SongDetectGroup objects
    """
    detectGen = detect(imagePath, show)
    while True:
        try:
            next(detectGen)
        except StopIteration as e:
            return e.value


def detect(imagePath: str,show: bool = False) -> Generator[int, None, list[SongDetectGroup]]:

    yield 0; # 0% progress

    if not modelReady:
        print("Model not ready. Please call prepare_model() first.")
        return []

    yield 20; # 20% progress

    # Detect
    results = model.predict(imagePath)


    yield 70; # 70% progress
    
    formattedResults : list[CustomDetect] = []
    for result in results:
        for box in result.boxes:
            formattedResults.append(CustomDetect(box, result))


        
    for result in formattedResults:
        if show:
            windowName = "image" + str(result.bounds.left) + str(result.bounds.top)
            cv.imshow(windowName, result.image)
            
            

    yield 90; # 90% progress
        

    # Group results
    songDetectGroups = groupCustomDetect(formattedResults)


    if show:
        image = cv.imread(imagePath)
        print(image.shape, results[0].orig_img.shape)
        renderResults(image, songDetectGroups, strokeWidth=3, fontSize=2)
        cv.imshow("Input-image", image)

    if show:
        cv.waitKey()

    yield 100
    return songDetectGroups


def renderResults(image, results: list[SongDetectGroup], strokeWidth=2, fontSize=1):
    for group in results:
        for box in [group.title, group.data, group.sheet]:
            if not box:
                continue
            bounds = box.bounds
            cv.rectangle(image, (int(bounds.left), int(bounds.top)),
                        (int(box.bounds.left + box.bounds.width), int(box.bounds.top + box.bounds.height)), (255, 0, 0), strokeWidth)
            cv.putText(image, f"{box.label}",
                        (int(box.bounds.left), int(box.bounds.top) - 10),
                        cv.FONT_HERSHEY_PLAIN, fontSize, (255, 0, 0), strokeWidth)

    return image
    


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

        # Convert YOLO results to CustomDetect objects
        formattedResults = []
        for result in results:
            for box in result.boxes:
                formattedResults.append(CustomDetect(box, result))

        # Group results
        songDetectGroups = groupCustomDetect(formattedResults)

        # Draw results
        frame = renderResults(frame, songDetectGroups)

        # Display the resulting frame
        cv.imshow('frame', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()