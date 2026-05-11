import json
class Sheet:
    def __init__ (self, title: str, data: str, inputImagePath: str, cropedImageData):
        self.title = title
        self.data = data
        self.inputImagePath = inputImagePath
        self.cropedImageData = cropedImageData

    def to_json(self):
        return {
            "title": self.title,
            "data": self.data,
            "inputImagePath": self.inputImagePath,
        }