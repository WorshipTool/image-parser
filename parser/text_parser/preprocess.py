import cv2
import numpy as np
from pathlib import Path

def _odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1

def preprocess(input_img, debug: bool = False) -> np.ndarray:
    if len(input_img.shape) == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        img = input_img.copy()

    h, w = img.shape[:2]
    s = min(h, w)

    # --- RELATIVE PARAMETERS (tune these ratios if needed) ---
    denoise_h = int(max(8, min(30, s * 0.012)))        # ~1.2% of short side
    sigma = float(max(10.0, s * 0.03))                 # ~3% of short side (illumination flatten)
    block = _odd(int(max(31, s * 0.05)))               # ~5% of short side, odd, min 31
    C = int(max(2, s * 0.004))                         # ~0.4% of short side, min 2
    ksize = int(max(1, round(s * 0.002)))              # ~0.2% of short side, min 1
    # --------------------------------------------------------

    # 1) denoise
    img = cv2.fastNlMeansDenoising(
        img, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21
    )

    # 2) flatten illumination
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    img = cv2.divide(img, bg, scale=255)

    # 3) adaptive threshold
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        C,
    )

    # 4) cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=1)

    if debug:
        import uuid
        temp_dir = Path(__file__).parent.parent.parent / "temp" / "ocr_preprocessing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        debug_path = temp_dir / f"preprocessed_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(debug_path), img)
        print(f"  ✓ Preprocessed image saved: {debug_path}")

        # optional: print params for quick tuning
        print(f"  params: s={s}, denoise_h={denoise_h}, sigma={sigma:.1f}, block={block}, C={C}, k={ksize}")

    return img