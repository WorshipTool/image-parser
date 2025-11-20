from .word import Word

class Line:
    def __init__(self, centerY : float, chordLinePossibility: float, words: list[Word]):
        self.centerY = centerY
        self.chordLinePossibility = chordLinePossibility
        self.words = words  
    
    def __str__(self):
        lineString = ""
        for word in self.words:
            lineString += word.text + " "
        return lineString
