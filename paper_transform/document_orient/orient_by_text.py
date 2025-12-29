"""
Text-based orientation detection for documents
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional, Literal




from typing import Optional, Tuple
import numpy as np

def orient_by_text(image: np.ndarray) -> Optional[Tuple[np.ndarray, int, float]]:
    """
    Try rotations (0/90/180/270) and pick the one with the best OCR-based score.
    Returns (best_img, best_angle_degrees, best_score). None if no text found anywhere.
    """
    import pytesseract
    import cv2
    from pytesseract import Output

    PSM = 6                
    GOOD_CONF = 60.0
    GOOD_LEN = 2
    GOOD_WEIGHT = 2.5      

    def score_for(img_bgr: np.ndarray) -> float:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(
            rgb,
            output_type=Output.DICT,
            config=f"--oem 3 --psm {PSM}"
        )

        confs = []
        good = 0
        any_text = 0

        for c, txt in zip(data["conf"], data["text"]):
            if c == "-1":
                continue
            txt = (txt or "").strip()
            if not txt:
                continue

            any_text += 1
            cf = float(c)
            confs.append(cf)

            if cf >= GOOD_CONF and len(txt) >= GOOD_LEN:
                good += 1

        if not confs or any_text == 0:
            return float("-inf")

        avg = sum(confs) / len(confs)
        return avg + GOOD_WEIGHT * good

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
        print(f"Rotation {angle}°: score = {score:.2f}")
        if score > best_score:
            best_score = score
            best_angle = angle
            best_img = img_rot

    if best_angle is None or best_score == float("-inf"):
        return None

    return best_img, best_angle, best_score