from common.bounds import Bounds

class Word:
    def __init__ (self, text: str, bounds: Bounds): 
        self.text = text
        self.bounds = bounds

        