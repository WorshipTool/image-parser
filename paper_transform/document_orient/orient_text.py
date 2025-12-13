"""
Text-based orientation detection for documents
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional, Literal
import logging

from .ocr_engines import create_ocr_engine, OCREngine

# Import fallback orientation from parent module
try:
    from ..orient import auto_orient as geometric_orient
    GEOMETRIC_FALLBACK_AVAILABLE = True
except ImportError:
    GEOMETRIC_FALLBACK_AVAILABLE = False

logger = logging.getLogger(__name__)


def _rotate_image(image: np.ndarray, angle: Literal[0, 90, 180, 270]) -> np.ndarray:
    """
    Rotate image by specified angle.

    Args:
        image: Input image
        angle: Rotation angle (0, 90, 180, or 270 degrees)

    Returns:
        Rotated image
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError(f"Invalid rotation angle: {angle}. Must be 0, 90, 180, or 270")


def _detect_text_orientation(image: np.ndarray) -> Optional[str]:
    """
    Detect if text runs horizontally or vertically using Tesseract's layout analysis.

    Args:
        image: Input image in BGR format

    Returns:
        'horizontal' if text runs left-to-right, 'vertical' if top-to-bottom, None if unknown
    """
    try:
        import pytesseract
        import cv2
        from pytesseract import Output

        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes using --psm 3
        data = pytesseract.image_to_data(rgb, output_type=Output.DICT, config='--psm 3')

        # Filter out empty detections and get lines
        lines = []
        current_line = []
        prev_line_num = -1

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Only confident detections
                text = data['text'][i].strip()
                if text:  # Non-empty text
                    line_num = data['line_num'][i]
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                    if line_num != prev_line_num and current_line:
                        lines.append(current_line)
                        current_line = []

                    current_line.append((x, y, w, h))
                    prev_line_num = line_num

        if current_line:
            lines.append(current_line)

        if len(lines) < 2:  # Not enough lines to determine orientation
            return None

        # Analyze line layout
        # For each line, check if words are arranged horizontally or vertically
        horizontal_evidence = 0
        vertical_evidence = 0

        for line in lines:
            if len(line) < 2:
                continue

            # Calculate line span in X and Y directions
            xs = [x for x, y, w, h in line]
            ys = [y for x, y, w, h in line]

            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)

            # If words in a line span more horizontally, it's horizontal text
            # If they span more vertically, it's vertical text
            if x_span > y_span * 2:  # Horizontal layout
                horizontal_evidence += 1
            elif y_span > x_span * 2:  # Vertical layout
                vertical_evidence += 1

        # Decide based on evidence
        if horizontal_evidence > vertical_evidence * 1.5:
            return 'horizontal'
        elif vertical_evidence > horizontal_evidence * 1.5:
            return 'vertical'
        else:
            return None  # Unclear

    except Exception:
        return None


def _calculate_text_score(text: str, debug: bool = False) -> float:
    """
    Calculate readability score for extracted text.

    The score is based on:
    - Length of text (more text = better)
    - Purity: ratio of alphanumeric characters to total characters
    - Word ratio: ratio of actual words to total tokens (detects gibberish)

    This scoring works well for documents with chords, lyrics, and mixed content.

    Args:
        text: Extracted text from OCR
        debug: If True, print detailed scoring info

    Returns:
        Score value (higher is better)
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    # Remove excessive whitespace but keep structure
    cleaned_text = ' '.join(text.split())

    # Count total characters
    total_chars = len(cleaned_text)

    if total_chars == 0:
        return 0.0

    # Count "good" characters: letters, numbers, common punctuation, diacritics
    # Include Czech and other Latin Extended characters
    # Pattern includes: A-Z, a-z, 0-9, spaces, punctuation, accented characters
    good_chars = re.findall(
        r'[A-Za-z0-9\s\.,;:!?\-\'\"\(\)\[\]/\u00C0-\u017F\u0100-\u024F]',
        cleaned_text
    )
    good_char_count = len(good_chars)

    # Calculate purity ratio
    purity = good_char_count / total_chars

    # Penalize very low purity (lots of unreadable/garbage characters)
    if purity < 0.3:
        purity = purity * 0.5  # Heavy penalty for mostly garbage

    # Bonus for high purity
    if purity > 0.9:
        purity = purity * 1.1  # Small bonus for very clean text

    # Calculate word ratio to detect gibberish
    # Split by whitespace and count tokens with reasonable length (2+ chars)
    tokens = [t.strip('.,;:!?\-\'\"()[]') for t in cleaned_text.split()]
    tokens = [t for t in tokens if len(t) >= 2]

    word_ratio = 1.0  # Default to 1.0 if no tokens
    if tokens:
        # Count tokens that look like words (mostly letters, min 2 chars)
        word_like = sum(1 for t in tokens if re.match(r'^[A-Za-z\u00C0-\u017F\u0100-\u024F]{2,}', t))
        word_ratio = word_like / len(tokens)

        # Heavily penalize low word ratio (indicates upside down or gibberish)
        if word_ratio < 0.3:
            purity = purity * 0.3  # 70% penalty for mostly gibberish

    # Calculate final score
    # Length * purity gives higher scores to longer, cleaner text
    score = total_chars * purity

    if debug:
        print(f"    [Scoring] len={total_chars}, purity={purity:.2f}, word_ratio={word_ratio:.2f}, score={score:.2f}")

    return score


def orient_by_text(
    image_bgr: np.ndarray,
    ocr_engine: str = "tesseract",
    ocr_kwargs: Optional[dict] = None,
    min_score_threshold: float = 5.0,
    debug: bool = False
) -> np.ndarray:
    """
    Automatically orient document based on text detection using OCR.

    This function tries all four possible rotations (0°, 90°, 180°, 270°),
    runs OCR on each, and selects the rotation with the best text readability score.

    If OCR fails for all rotations, falls back to geometric orientation
    based on aspect ratio.

    Args:
        image_bgr: Input warped document image in BGR format
        ocr_engine: OCR engine to use ('tesseract' or 'easyocr')
        ocr_kwargs: Optional keyword arguments for OCR engine initialization
        min_score_threshold: Minimum score to trust OCR result. If all rotations
                             score below this, fall back to geometric orientation.
        debug: If True, print debug information about scores

    Returns:
        Oriented image in BGR format with text right-side up

    Raises:
        ValueError: If image is None or empty
        ImportError: If OCR engine is not available and no fallback exists

    Example:
        >>> from paper_detection import PaperDetector
        >>> from paper_transform import warp_paper
        >>> from paper_detection.document_orient import orient_by_text
        >>>
        >>> # Detect and warp
        >>> detector = PaperDetector()
        >>> corners = detector.detect(image)
        >>> warped = warp_paper(image, corners)
        >>>
        >>> # Orient based on text
        >>> oriented = orient_by_text(warped)
        >>> cv2.imwrite('result.jpg', oriented)

    Note:
        OCR can be computationally expensive. For documents without text,
        consider using the faster geometric orientation from paper_transform.auto_orient.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is None or empty")

    # Initialize OCR engine
    ocr_kwargs = ocr_kwargs or {}
    try:
        ocr = create_ocr_engine(ocr_engine, **ocr_kwargs)
    except ImportError as e:
        # OCR not available, try geometric fallback immediately
        logger.warning(f"OCR engine not available: {e}")
        return _fallback_to_geometric(image_bgr)

    # Test all four rotations
    rotations = [0, 90, 180, 270]
    scores = {}

    if debug:
        print("\n" + "=" * 60)
        print("Text Orientation Detection - Debug Info")
        print("=" * 60)

    for angle in rotations:
        # Rotate image physically
        rotated = _rotate_image(image_bgr, angle)

        if debug:
            print(f"\nRotation {angle:3d}°:")
            print(f"  Image rotated: {rotated.shape[1]}x{rotated.shape[0]} px")

        # Extract text via OCR (Tesseract does NOT auto-rotate)
        try:
            text = ocr.extract_text(rotated)

            # Also get confidence score if available (Tesseract only)
            confidence = 0.0
            if ocr_engine == "tesseract":
                try:
                    import pytesseract
                    from pytesseract import Output
                    data = pytesseract.image_to_data(rotated, output_type=Output.DICT)
                    # Average confidence of detected text
                    confidences = [int(c) for c in data['conf'] if int(c) > 0]
                    if confidences:
                        confidence = sum(confidences) / len(confidences)
                except Exception:
                    confidence = 0.0

        except Exception as e:
            logger.warning(f"OCR failed for {angle}° rotation: {e}")
            text = ""
            confidence = 0.0

        # Calculate base score based on text quality
        score = _calculate_text_score(text, debug=debug)

        # Apply confidence multiplier (0-100 range → 0-1.5 multiplier)
        # High confidence (80+): 1.2x boost
        # Medium confidence (50-80): 1.0x normal
        # Low confidence (<50): 0.7x penalty
        if confidence > 0:
            if confidence >= 80:
                confidence_mult = 1.2
            elif confidence >= 50:
                confidence_mult = 1.0
            else:
                confidence_mult = 0.7
            score = score * confidence_mult

            if debug:
                print(f"  OCR confidence: {confidence:.1f}% (multiplier: {confidence_mult:.2f}x)")

        # Detect if text runs horizontally or vertically
        text_dir = _detect_text_orientation(rotated)

        # HEAVILY penalize vertical text
        # We want text to run horizontally (left-to-right)
        if text_dir == 'vertical':
            score = score * 0.05  # 95% penalty for vertical text
            if debug:
                print(f"  Text direction: VERTICAL (penalized)")
        elif text_dir == 'horizontal':
            if debug:
                print(f"  Text direction: horizontal (good)")
        elif debug:
            print(f"  Text direction: unknown")

        scores[angle] = score

        if debug:
            text_preview = text[:80].replace('\n', ' ') if text else "(no text)"
            char_count = len(text.strip())
            print(f"  Characters extracted: {char_count}")
            print(f"  Final score: {score:7.2f}")
            print(f"  Text preview: {text_preview}...")

    # Find best rotation
    best_angle = max(scores, key=scores.get)
    best_score = scores[best_angle]

    if debug:
        print("\n" + "-" * 60)
        print(f"Best rotation: {best_angle}° (score: {best_score:.2f})")
        print("=" * 60 + "\n")

    # Check if score is reliable
    if best_score < min_score_threshold:
        logger.info(
            f"OCR score too low ({best_score:.2f} < {min_score_threshold}), "
            "falling back to geometric orientation"
        )
        return _fallback_to_geometric(image_bgr)

    # Return best rotation
    return _rotate_image(image_bgr, best_angle)


def _fallback_to_geometric(image_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback to geometric orientation when OCR fails.

    Args:
        image_bgr: Input image

    Returns:
        Oriented image using aspect ratio heuristic
    """
    if GEOMETRIC_FALLBACK_AVAILABLE:
        logger.info("Using geometric orientation fallback")
        # Use auto_geometric to avoid circular recursion
        return geometric_orient(image_bgr, orientation="auto_geometric", use_ocr=False)
    else:
        # No fallback available, return original
        logger.warning("No geometric fallback available, returning original image")
        return image_bgr


def batch_orient_text(
    images: list,
    ocr_engine: str = "tesseract",
    ocr_kwargs: Optional[dict] = None,
    debug: bool = False
) -> list:
    """
    Orient multiple document images based on text detection.

    This is a convenience function for processing multiple images.
    The OCR engine is initialized once and reused for all images.

    Args:
        images: List of input images in BGR format
        ocr_engine: OCR engine to use ('tesseract' or 'easyocr')
        ocr_kwargs: Optional keyword arguments for OCR engine initialization
        debug: If True, print debug information

    Returns:
        List of oriented images

    Example:
        >>> from paper_detection.document_orient import batch_orient_text
        >>> images = [cv2.imread(f'page_{i}.jpg') for i in range(5)]
        >>> oriented_images = batch_orient_text(images)
    """
    oriented = []

    for i, image in enumerate(images):
        if debug:
            print(f"\nProcessing image {i+1}/{len(images)}...")

        try:
            result = orient_by_text(
                image,
                ocr_engine=ocr_engine,
                ocr_kwargs=ocr_kwargs,
                debug=debug
            )
            oriented.append(result)
        except Exception as e:
            logger.error(f"Failed to orient image {i}: {e}")
            oriented.append(image)  # Keep original on failure

    return oriented
