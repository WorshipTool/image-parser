"""
Type definitions for paper detection
"""

import numpy as np
from numpy.typing import NDArray

# Shape: [4, 2] - 4 corners with (x, y) coordinates
Corners = NDArray[np.float32]
