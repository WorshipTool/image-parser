import os
import cv2 as cv

import song_detection;

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "yolo8best.pt")
song_detection.prepare_model(model_path)

# load imagepath from argument
imagePath = ""
if(len(os.sys.argv) > 1):
    imagePath = os.sys.argv[1]
else:
    print("Please provide image path as argument.")
    exit(0)

# Detect
results = song_detection.detect(imagePath, show=False)

inputImage = cv.imread(imagePath)
# # Draw
song_detection.renderResults(inputImage,results, strokeWidth=3, fontSize=2)

# Show
cv.imshow("Result", inputImage)
cv.waitKey(0)
cv.destroyAllWindows()



