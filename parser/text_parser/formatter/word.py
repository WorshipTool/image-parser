from common.bounds import Bounds

class Word:
    def __init__ (self, text: str, bounds: Bounds, confidence: float): 
        self.text = text
        self.bounds = bounds,
        self.confidence = confidence

        