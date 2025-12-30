import os
import sys
import cv2 as cv

# Add parent directory to path to import sheet_detection as module
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.insert(0, parent_directory)

import sheet_detection  # Auto-initializes model on import

# load imagepath from argument
imagePath = ""
if(len(os.sys.argv) > 1):
    imagePath = os.sys.argv[1]
else:
    print("Please provide image path as argument.")
    exit(0)

# Detect
results = sheet_detection.detect_simple(imagePath, show=False)

inputImage = cv.imread(imagePath)
# Draw
sheet_detection.renderResults(inputImage, results, strokeWidth=3, fontSize=2)

# Show
cv.imshow("Result", inputImage)
cv.waitKey(0)
cv.destroyAllWindows()



