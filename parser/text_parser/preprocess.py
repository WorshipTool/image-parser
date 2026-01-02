import cv2
import numpy as np
from pathlib import Path

def preprocess(img, debug: bool = False) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # upscale – lepší pro malé písmo
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # odšumění – mírnější (median 3 často žere tahy)
    gray = cv2.medianBlur(gray, 1)  # nebo to klidně smaž

    # adaptive threshold – mírnější pro tenké tahy
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        71,  # větší block
        6,   # menší C = méně ztenčení písma
    )

    # morfologie – u malého textu radši vypnout
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

    if debug:
        import uuid
        temp_dir = Path(__file__).parent.parent.parent / "temp" / "ocr_preprocessing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        debug_path = temp_dir / f"preprocessed_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(debug_path), bin_img)
        print(f"  ✓ Preprocessed image saved: {debug_path}")

    return bin_img