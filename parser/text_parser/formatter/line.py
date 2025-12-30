from common.bounds import Bounds
from .word import Word

class Line:
    def __init__(self, centerY : float, chordLinePossibility: float, words: list[Word]):
        self.centerY = centerY
        self.chordLinePossibility = chordLinePossibility
        self.words = words
        self.avgConfidence = sum([word.confidence for word in words]) / len(words) if words else 0.0

    @property
    def bounds(self) -> Bounds:
        """
        Calculate bounds dynamically from all words in the line
        """
        if not self.words:
            return Bounds(0, 0, 0, 0)

        left = min(word.bounds.left for word in self.words)
        right = max(word.bounds.left + word.bounds.width for word in self.words)
        top = min(word.bounds.top for word in self.words)
        bottom = max(word.bounds.top + word.bounds.height for word in self.words)

        return Bounds(left, top, right - left, bottom - top)

    def __str__(self):
        lineString = ""
        for word in self.words:
            lineString += word.text + " "
        return lineString
