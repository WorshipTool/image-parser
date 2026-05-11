import cv2
import numpy as np
from pathlib import Path

def _resize_to_short_side(img: np.ndarray, target_short: int = 1200):
    h, w = img.shape[:2]
    s = min(h, w)
    if s == target_short:
        return img, 1.0
    scale = target_short / float(s)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=interp)
    return resized, scale

def remove_horizontal_lines(img: np.ndarray) -> np.ndarray:

    min_len = max(60, img.shape[1] // 8)

    # 3) invertuj JEN PRO MORFOLOGII
    bw_inv = cv2.bitwise_not(img)

    # 4) detekuj horizontální čáry
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    lines = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, kernel)

    # 5) odeber čáry (stále v inverted prostoru)
    cleaned_inv = cv2.bitwise_and(bw_inv, cv2.bitwise_not(lines))

    # 6) VRAŤ zpět do normálu
    cleaned = cv2.bitwise_not(cleaned_inv)
    return cleaned

def preprocess(input_img, debug: bool = False, target_short: int = 1200) -> np.ndarray:
    if len(input_img.shape) == 3:
        gray0 = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        gray0 = input_img.copy()

    gray, scale = _resize_to_short_side(gray0, target_short=target_short)


    # FIXED parameters (stable across scales)
    denoise_h = 12
    sigma = 35.0   # back down (less edge boosting)
    block = 81     # moderate local window
    C = 20          # more tolerant -> fewer background artefacts
    ksize = 3      # a bit stronger cleanup

    gray = cv2.fastNlMeansDenoising(
        gray, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21
    )

    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gray = cv2.divide(gray, bg, scale=255)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, C
    )



    bw = remove_horizontal_lines(bw)

    if bw.shape != gray0.shape:
        bw = cv2.resize(bw, (gray0.shape[1], gray0.shape[0]), interpolation=cv2.INTER_NEAREST)

    if debug:
        import uuid
        temp_dir = Path(__file__).parent.parent.parent / "temp" / "ocr_preprocessing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        debug_path = temp_dir / f"preprocessed_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(debug_path), bw)
        print(f"  ✓ Preprocessed image saved: {debug_path}")
        print(f"  params: target_short={target_short}, scale={scale:.3f}, sigma={sigma}, block={block}, C={C}, ksize={ksize}")

    return bw