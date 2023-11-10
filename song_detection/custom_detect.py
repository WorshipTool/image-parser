class CustomDetect:
    def __init__(self, input):
        self.label : str = input["label"].split(" ")[0]
        self.confidence : float = input["conf"].item()
        self.image = input["im"]
        