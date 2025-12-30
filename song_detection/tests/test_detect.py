import os
import sys

# Add parent directory to path
current_directory = os.path.dirname(os.path.abspath(__file__))
song_detection_dir = os.path.dirname(current_directory)
parent_directory = os.path.dirname(song_detection_dir)
sys.path.insert(0, parent_directory)

from song_detection import detect_simple, prepare_model

model_path = os.path.join(parent_directory, "yolo8best.pt")
prepare_model(model_path)

class TestClass:
    def test_detect1(self):
        test_image = os.path.join(current_directory, "images", "img1.jpeg")
        results = detect_simple(test_image)
        assert len(results) >= 3

    def test_detect2(self):
        test_image = os.path.join(current_directory, "images", "img2.jpg")
        results = detect_simple(test_image)
        assert len(results) >= 3