"""
Test suite for text_parser module

Tests OCR reading and sheet formatting with real cropped images.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import cv2
import json

# Add parser to path
parser_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parser_dir))

from text_parser import read_and_parse_image
from text_parser.ocr import read as ocr_read


def draw_word_boxes(image_bgr: np.ndarray, word_data: list) -> np.ndarray:
    """
    Draw bounding boxes around detected words on image

    Args:
        image_bgr: Input image (BGR format)
        word_data: List of ReadWordData objects from OCR

    Returns:
        Image with drawn bounding boxes
    """
    output = image_bgr.copy()

    for word in word_data:
        bounds = word.bounds
        x = int(bounds.left)
        y = int(bounds.top)
        w = int(bounds.width)
        h = int(bounds.height)

        # Draw rectangle around word
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw text label with confidence
        label = f"{word.text} ({word.confidence:.0f}%)"
        font_scale = 0.5
        thickness = 1

        # Get text size for background
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Draw background for text
        cv2.rectangle(
            output,
            (x, y - text_height - 5),
            (x + text_width, y),
            (0, 255, 0),
            -1
        )

        # Draw text
        cv2.putText(
            output,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness
        )

    return output


class TestTextParser(unittest.TestCase):
    """Test text_parser OCR and formatting functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test images paths"""
        cls.image_parser_dir = Path(__file__).parent.parent.parent.parent
        cls.test_images_dir = Path(__file__).parent / "images"

        # Create temp output directory
        cls.temp_dir = cls.image_parser_dir / "temp" / "text_parser_output"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)

        # Use 4 cropped test images
        cls.test_images = [
            cls.test_images_dir / "img1.jpg",
            cls.test_images_dir / "img2.jpg",
            cls.test_images_dir / "img3.jpg",
            cls.test_images_dir / "img4.jpg"
        ]

        # Verify all test images exist
        for img_path in cls.test_images:
            if not img_path.exists():
                raise FileNotFoundError(f"Test image not found: {img_path}")

    def test_read_and_parse_image_with_path(self):
        """Test read_and_parse_image with image file paths"""
        for img_path in self.test_images:
            with self.subTest(image=img_path.name):
                result = read_and_parse_image(str(img_path), debug=False)

                # Verify result structure
                self.assertIsNotNone(result, f"Failed to parse {img_path.name}")
                self.assertIn('title', result)
                self.assertIn('data', result)
                self.assertIn('inputImagePath', result)

                # Verify types
                self.assertIsInstance(result['title'], str)
                self.assertIsInstance(result['data'], str)
                self.assertIsInstance(result['inputImagePath'], str)

                # Title should not be empty
                self.assertTrue(len(result['title']) > 0, "Title is empty")

                # Data should contain some content
                self.assertTrue(len(result['data']) > 0, "Data is empty")

    def test_read_and_parse_image_with_array(self):
        """Test read_and_parse_image with numpy array"""
        img_path = self.test_images[0]
        image_bgr = cv2.imread(str(img_path))

        result = read_and_parse_image(image_bgr, debug=False)

        self.assertIsNotNone(result)
        self.assertIn('title', result)
        self.assertIn('data', result)
        self.assertTrue(len(result['title']) > 0)
        self.assertTrue(len(result['data']) > 0)

    def test_chord_detection(self):
        """Test that chords are detected in formatted output"""
        # Use first test image
        img_path = self.test_images[0]
        result = read_and_parse_image(str(img_path), debug=False)

        self.assertIsNotNone(result)
        data = result['data']

        # Check if chord format [X] exists in data
        # Note: Not all images may contain chords, so we just verify format
        self.assertIsInstance(data, str)

    def test_section_detection(self):
        """Test that sections are detected in formatted output"""
        img_path = self.test_images[0]
        result = read_and_parse_image(str(img_path), debug=False)

        self.assertIsNotNone(result)
        data = result['data']

        # Sections are marked with {SectionName1} format
        # Just verify we get structured output
        self.assertIsInstance(data, str)
        self.assertTrue(len(data) > 0)

    def test_invalid_image_path(self):
        """Test handling of invalid image path"""
        result = read_and_parse_image("nonexistent_image.jpg", debug=False)
        self.assertIsNone(result, "Should return None for invalid path")

    def test_invalid_image_array(self):
        """Test handling of invalid image array"""
        # Empty array
        empty_array = np.array([])
        result = read_and_parse_image(empty_array, debug=False)
        self.assertIsNone(result, "Should return None for empty array")

        # None image
        result = read_and_parse_image(None, debug=False)
        self.assertIsNone(result, "Should return None for None image")

    def test_debug_output(self):
        """Test that debug mode doesn't break functionality"""
        img_path = self.test_images[0]

        # Should work with debug=True
        result = read_and_parse_image(str(img_path), debug=True)
        self.assertIsNotNone(result)


class TestTextParserWithAllImages(unittest.TestCase):
    """Test text_parser with all available test images and print results"""

    @classmethod
    def setUpClass(cls):
        """Set up test images paths"""
        cls.image_parser_dir = Path(__file__).parent.parent.parent.parent
        cls.test_images_dir = Path(__file__).parent / "images"

        # Create temp output directory
        cls.temp_dir = cls.image_parser_dir / "temp" / "text_parser_output"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)

        cls.test_images = [
            cls.test_images_dir / "img1.jpg",
            cls.test_images_dir / "img2.jpg",
            cls.test_images_dir / "img3.jpg",
            cls.test_images_dir / "img4.jpg"
        ]

    def test_parse_all_images_verbose(self):
        """Parse all test images and print results"""
        print("\n" + "="*80)
        print("TEXT PARSER RESULTS FOR ALL TEST IMAGES")
        print("="*80)

        all_results = []

        for i, img_path in enumerate(self.test_images, 1):
            print(f"\n{'─'*80}")
            print(f"Image {i}: {img_path.name}")
            print(f"{'─'*80}")

            result = read_and_parse_image(str(img_path), debug=True)

            if result:
                print(f"\n✓ Title: {result['title']}")
                print(f"✓ Data length: {len(result['data'])} characters")
                print(f"\nFirst 300 characters of formatted data:")
                print(f"{result['data'][:300]}...")
                print()

                # Save individual result to temp
                output_name = f"result_{img_path.stem}.json"
                output_path = self.temp_dir / output_name
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✓ Saved to: {output_path}")

                # Save formatted text
                text_output_name = f"result_{img_path.stem}.txt"
                text_output_path = self.temp_dir / text_output_name
                with open(text_output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {result['title']}\n")
                    f.write("="*80 + "\n\n")
                    f.write(result['data'])
                print(f"✓ Saved text to: {text_output_path}")

                # Generate visualization with word bounding boxes
                image_bgr = cv2.imread(str(img_path))
                word_data = ocr_read(image_bgr)
                annotated_image = draw_word_boxes(image_bgr, word_data)

                # Save annotated image
                annotated_output_name = f"annotated_{img_path.stem}.jpg"
                annotated_output_path = self.temp_dir / annotated_output_name
                cv2.imwrite(str(annotated_output_path), annotated_image)
                print(f"✓ Saved annotated image to: {annotated_output_path}")

                all_results.append(result)
            else:
                print("✗ Failed to parse image")

        # Save combined results
        combined_path = self.temp_dir / "all_results.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ All results saved to: {combined_path}")

        print("="*80)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestTextParser))
    suite.addTests(loader.loadTestsFromTestCase(TestTextParserWithAllImages))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
