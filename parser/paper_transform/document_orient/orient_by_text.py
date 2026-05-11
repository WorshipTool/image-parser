"""
Text-based orientation detection for documents
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional, Literal




from typing import Optional, Tuple
import numpy as np

def orient_by_text(image: np.ndarray, debug: bool = False) -> Optional[Tuple[np.ndarray, int, float]]:
    """
    Try rotations (0/90/180/270) and pick the one with the best OCR-based score,
    but using image_to_string (faster).
    Returns (best_img, best_angle_degrees, best_score). None if no text found anywhere.
    """
    import pytesseract
    import cv2

    PSM = 11
    MAX_SIDE = 900

    # Optional speed: disable dictionaries (often faster for this "orientation" heuristic)
    EXTRA_CFG = "-c load_system_dawg=0 -c load_freq_dawg=0"
    CONFIG = f"--oem 3 --psm {PSM} {EXTRA_CFG}"
    # If you want language: add e.g.  LANG = "ces" and CONFIG = f"... -l {LANG} ..."
    # (but language can slow it a bit)

    def downscale(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def score_for(img_bgr: np.ndarray) -> float:
        img_small = downscale(img_bgr)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        # light normalization (cheap, helps sometimes)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        txt = pytesseract.image_to_string(gray, config=CONFIG)
        txt = (txt or "").strip()

        if not txt:
            return float("-inf")

        # score = how "text-like" the output is
        letters = sum(ch.isalpha() for ch in txt)
        digits = sum(ch.isdigit() for ch in txt)
        spaces = sum(ch.isspace() for ch in txt)
        total = len(txt)

        # penalize garbage symbols a bit
        symbols = total - letters - digits - spaces
        return (letters + 0.3 * digits) - 0.7 * symbols

    rotations = [
        (0, image),
        (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(image, cv2.ROTATE_180)),
        (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_angle = None
    best_score = float("-inf")
    best_img = None

    for angle, img_rot in rotations:
        score = score_for(img_rot)
        if debug:
            print(f"  Angle {angle}°: score = {score}")
        if score > best_score:
            best_score = score
            best_angle = angle
            best_img = img_rot

    if best_angle is None or best_score == float("-inf"):
        return None

    return best_img, best_angle, best_score