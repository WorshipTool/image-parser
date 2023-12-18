import cv2 as cv

import song_detection;

song_detection.prepare_model("yolo8best.pt")

song_detection.launchRealTimeDetection()