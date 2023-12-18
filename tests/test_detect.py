from song_detection import detect, prepare_model;

prepare_model("yolo8best.pt")

class TestClass:
    def test_detect1(self):
        results = detect("tests/images/img1.jpeg")
        assert len(results) >= 3

    def test_detect2(self):
        results = detect("tests/images/img2.jpg")
        assert len(results) >= 3