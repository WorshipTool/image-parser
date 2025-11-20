class Bounds:
    def __init__ (self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return f"Bounds({self.left}, {self.top}, {self.width}, {self.height})"

    def center(self) -> tuple[int, int]:
        return (self.left + self.width / 2, self.top + self.height / 2)
    
    def isInside(self, other, checkCenterOnly=False) -> bool:
        """
        Check if the current Bounds object is completely inside another Bounds object.

        Parameters:
        - other (Bounds): The other Bounds object to compare against.

        Returns:
        - bool: True if the current Bounds object is completely inside the other Bounds object, False otherwise.
        """


        if checkCenterOnly:
            center = self.center()
            return Bounds(center[0], center[1],0,0).isInside(other)
        
        return self.left >= other.left and self.left + self.width <= other.left + other.width and self.top >= other.top and self.top + self.height <= other.top + other.height
    
    def area(self):
        """
        Calculate the area of the Bounds object.

        Returns:
        - int: The area of the Bounds object.
        """
        return self.width * self.height