
from common.bounds import Bounds


class TestClass:
    def test_bounds1(self):
        bounds = Bounds(0, 0, 10, 10)
        assert bounds.left == 0 and bounds.top == 0 and bounds.width == 10 and bounds.height == 10
    
    def test_bounds2(self):
        bounds = Bounds(0, 0, 10, 10)
        assert bounds.area() == 100
    
    def test_bounds3(self):
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(2, 2, 4, 4)
        assert bounds2.isInside(bounds1)
    
    def test_bounds4(self):
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(2, 2, 10, 10)
        assert bounds2.isInside(bounds1, True)

    
    def test_bounds5(self):
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(6, 6, 10, 10)
        assert not bounds2.isInside(bounds1, True)
    
    def test_bounds6(self):
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(2, 2, 4, 4)
        assert not bounds1.isInside(bounds2)

    def test_bounds7(self):
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(2, 2, 10, 10)
        assert bounds1.isInside(bounds2, True)
    
    def test_bounds8(self):
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(-5,-1, 15, 4)
        assert not bounds1.isInside(bounds2, True)

    
    def test_bounds9(self):
        bounds1 = Bounds(0, 0, 10, 10)
        bounds2 = Bounds(-5,-1, 15, 4)
        assert bounds2.isInside(bounds1, True)
