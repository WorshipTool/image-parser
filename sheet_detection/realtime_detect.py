import os
import sys

# Add parent directory to path to import sheet_detection as module
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.insert(0, parent_directory)

import sheet_detection

model_path = os.path.join(parent_directory, "yolo8best.pt")
sheet_detection.prepare_model(model_path)

sheet_detection.launchRealTimeDetection()