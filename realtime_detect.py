import os

import song_detection;

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "yolo8best.pt")
song_detection.prepare_model(model_path)

song_detection.launchRealTimeDetection()