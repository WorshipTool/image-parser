from song_detection import detect, prepare_model;
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
model_path = os.path.join(parent_directory, "yolo8best.pt")
prepare_model(model_path)

class TestClass:
    def test_detect1(self):
        results = detect("tests/images/img1.jpeg")
        assert len(results) >= 3

    def test_detect2(self):
        results = detect("tests/images/img2.jpg")
        assert len(results) >= 3