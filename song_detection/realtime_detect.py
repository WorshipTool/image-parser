import os
import sys

# Add parent directory to path to import song_detection as module
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.insert(0, parent_directory)

import song_detection

model_path = os.path.join(parent_directory, "yolo8best.pt")
song_detection.prepare_model(model_path)

song_detection.launchRealTimeDetection()