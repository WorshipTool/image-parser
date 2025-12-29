"""
Text-based orientation detection for documents
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional, Literal

from .ocr_engines import create_ocr_engine




def orient_by_text(image: np.ndarray) -> Optional[Tuple[np.ndarray, int, float]]:
    """
    Try rotations (0/90/180/270) and pick the one with the best avg Tesseract confidence.
    Returns (best_angle_degrees, best_avg_conf). None if no text found anywhere.
    """
    import pytesseract
    import cv2
    from pytesseract import Output

    def avg_conf_for(img_bgr: np.ndarray) -> float:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(rgb, output_type=Output.DICT, config="--psm 3")
        confs = [float(c) for c in data["conf"] if c != "-1"]
        return (sum(confs) / len(confs)) if confs else float("-inf")

    rotations = [
        (0, image),
        (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(image, cv2.ROTATE_180)),
        (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_angle = None
    best_conf = float("-inf")
    best_img = None

    for angle, img_rot in rotations:
        conf = avg_conf_for(img_rot)
        if conf > best_conf:
            best_conf = conf
            best_angle = angle
            best_img = img_rot

    if best_angle is None or best_conf == float("-inf"):
        return None

    return best_img, best_angle, best_conf

