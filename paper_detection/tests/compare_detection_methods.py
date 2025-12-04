"""
Comparison evaluation script for OpenCV vs ML paper detection methods

This script compares three detection modes:
1. opencv: Traditional OpenCV-based detection
2. ml: Pure machine learning detection
3. ml_with_fallback: ML with OpenCV fallback

Generates comprehensive comparison statistics and visualizations.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper_detection import PaperDetector


@dataclass
class DetectionResult:
    """Store detection result for one image"""
    image_name: str
    success: bool
    corners: Optional[np.ndarray]
    avg_error: float
    max_error: float
    per_corner_errors: List[float]
    detection_time_ms: float


@dataclass
class MethodStatistics:
    """Statistics for one detection method"""
    method_name: str
    total_images: int
    successful_detections: int
    failed_detections: int
    success_rate: float
    avg_pixel_error: float
    median_pixel_error: float
    p95_pixel_error: float
    max_pixel_error: float
    avg_detection_time_ms: float


class DetectionMethodComparator:
    """Compare different paper detection methods"""

    def __init__(
        self,
        images_dir: Path,
        ground_truth_path: Path,
        output_dir: Path,
        ml_model_path: Optional[str] = None
    ):
        """
        Initialize comparator.

        Args:
            images_dir: Directory containing test images
            ground_truth_path: Path to ground truth JSON file
            output_dir: Directory for output visualizations
            ml_model_path: Optional path to ML model file
        """
        self.images_dir = images_dir
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir
        self.ml_model_path = ml_model_path

        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize detectors
        self.detectors = self._init_detectors()

    def _init_detectors(self) -> Dict[str, PaperDetector]:
        """Initialize all detection methods"""
        detectors = {
            'opencv': PaperDetector(detection_mode='opencv')
        }

        # Try to initialize ML detectors
        try:
            detectors['ml'] = PaperDetector(
                detection_mode='ml',
                ml_model_path=self.ml_model_path
            )
            detectors['ml_with_fallback'] = PaperDetector(
                detection_mode='ml_with_fallback',
                ml_model_path=self.ml_model_path
            )
            print("✓ ML detectors initialized successfully")
        except (ImportError, FileNotFoundError) as e:
            print(f"⚠ ML detectors not available: {e}")
            print("  Only OpenCV method will be evaluated")

        return detectors

    def _compute_corner_error(
        self,
        gt_corners: np.ndarray,
        detected_corners: np.ndarray
    ) -> Tuple[List[float], int]:
        """
        Compute corner-wise errors with best rotation matching.

        Returns:
            (errors for each corner, best rotation index)
        """
        best_rotation = 0
        best_total_error = float('inf')
        best_errors = []

        # Try all 4 rotations
        for rotation in range(4):
            rotated_corners = np.roll(detected_corners, rotation, axis=0)
            rotation_errors = []
            total_error = 0

            for i in range(4):
                distance = np.linalg.norm(gt_corners[i] - rotated_corners[i])
                rotation_errors.append(distance)
                total_error += distance

            if total_error < best_total_error:
                best_total_error = total_error
                best_rotation = rotation
                best_errors = rotation_errors

        return best_errors, best_rotation

    def _detect_with_timing(
        self,
        detector: PaperDetector,
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect corners and measure time.

        Returns:
            (detected corners, time in milliseconds)
        """
        import time
        start = time.time()
        corners = detector.detect(image)
        elapsed_ms = (time.time() - start) * 1000
        return corners, elapsed_ms

    def evaluate_single_image(
        self,
        image_name: str,
        method_name: str
    ) -> DetectionResult:
        """
        Evaluate single image with one detection method.

        Args:
            image_name: Name of image file
            method_name: Detection method ('opencv', 'ml', 'ml_with_fallback')

        Returns:
            DetectionResult object
        """
        # Load ground truth
        gt_data = self.ground_truth[image_name]
        tolerance = gt_data["tolerance_pixels"]

        # Load image
        image_path = self.images_dir / image_name
        image = cv2.imread(str(image_path))

        # Convert relative corners to absolute pixels
        h, w = image.shape[:2]
        gt_corners_relative = np.array(gt_data["corners"], dtype=np.float32)
        gt_corners = gt_corners_relative.copy()
        gt_corners[:, 0] *= w
        gt_corners[:, 1] *= h

        # Detect with timing
        detector = self.detectors[method_name]
        detected_corners, detection_time = self._detect_with_timing(detector, image)

        # Check if detection succeeded
        if detected_corners is None:
            return DetectionResult(
                image_name=image_name,
                success=False,
                corners=None,
                avg_error=float('inf'),
                max_error=float('inf'),
                per_corner_errors=[float('inf')] * 4,
                detection_time_ms=detection_time
            )

        # Compute errors with best rotation
        errors, best_rotation = self._compute_corner_error(gt_corners, detected_corners)
        avg_error = np.mean(errors)
        max_error = np.max(errors)

        # Check if within tolerance
        success = max_error <= tolerance

        return DetectionResult(
            image_name=image_name,
            success=success,
            corners=detected_corners,
            avg_error=avg_error,
            max_error=max_error,
            per_corner_errors=errors,
            detection_time_ms=detection_time
        )

    def evaluate_all_methods(self) -> Dict[str, List[DetectionResult]]:
        """
        Evaluate all detection methods on all images.

        Returns:
            Dictionary mapping method name to list of results
        """
        all_results = {method: [] for method in self.detectors.keys()}

        print(f"\n{'='*70}")
        print("Evaluating all detection methods...")
        print(f"{'='*70}\n")

        image_names = sorted(self.ground_truth.keys())

        for i, image_name in enumerate(image_names, 1):
            print(f"[{i}/{len(image_names)}] Processing {image_name}...")

            for method_name in self.detectors.keys():
                result = self.evaluate_single_image(image_name, method_name)
                all_results[method_name].append(result)

                # Print result
                if result.success:
                    print(f"  ✓ {method_name:20s}: avg={result.avg_error:6.2f}px, "
                          f"max={result.max_error:6.2f}px, time={result.detection_time_ms:6.1f}ms")
                else:
                    print(f"  ✗ {method_name:20s}: FAILED (time={result.detection_time_ms:6.1f}ms)")
            print()

        return all_results

    def compute_statistics(
        self,
        results: List[DetectionResult],
        method_name: str
    ) -> MethodStatistics:
        """
        Compute statistics for one method.

        Args:
            results: List of detection results
            method_name: Name of the method

        Returns:
            MethodStatistics object
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        total = len(results)
        success_count = len(successful_results)
        fail_count = len(failed_results)
        success_rate = success_count / total if total > 0 else 0.0

        # Compute error statistics (only for successful detections)
        if successful_results:
            avg_errors = [r.avg_error for r in successful_results]
            avg_pixel_error = np.mean(avg_errors)
            median_pixel_error = np.median(avg_errors)
            p95_pixel_error = np.percentile(avg_errors, 95)
            max_pixel_error = np.max(avg_errors)
        else:
            avg_pixel_error = float('inf')
            median_pixel_error = float('inf')
            p95_pixel_error = float('inf')
            max_pixel_error = float('inf')

        # Compute average detection time (all images)
        avg_time = np.mean([r.detection_time_ms for r in results])

        return MethodStatistics(
            method_name=method_name,
            total_images=total,
            successful_detections=success_count,
            failed_detections=fail_count,
            success_rate=success_rate,
            avg_pixel_error=avg_pixel_error,
            median_pixel_error=median_pixel_error,
            p95_pixel_error=p95_pixel_error,
            max_pixel_error=max_pixel_error,
            avg_detection_time_ms=avg_time
        )

    def generate_comparison_table(
        self,
        all_results: Dict[str, List[DetectionResult]]
    ) -> str:
        """
        Generate markdown comparison table.

        Args:
            all_results: Results for all methods

        Returns:
            Markdown table as string
        """
        # Compute statistics for each method
        stats = {}
        for method_name, results in all_results.items():
            stats[method_name] = self.compute_statistics(results, method_name)

        # Create markdown table
        table = []
        table.append("# Paper Detection Method Comparison\n")
        table.append("## Overall Statistics\n")
        table.append("| Method | Success Rate | Avg Error (px) | Median Error (px) | P95 Error (px) | Max Error (px) | Avg Time (ms) |")
        table.append("|--------|--------------|----------------|-------------------|----------------|----------------|---------------|")

        for method_name in ['opencv', 'ml', 'ml_with_fallback']:
            if method_name not in stats:
                continue

            s = stats[method_name]
            table.append(
                f"| {s.method_name:18s} | "
                f"{s.success_rate*100:6.1f}% ({s.successful_detections}/{s.total_images}) | "
                f"{s.avg_pixel_error:8.2f} | "
                f"{s.median_pixel_error:10.2f} | "
                f"{s.p95_pixel_error:9.2f} | "
                f"{s.max_pixel_error:9.2f} | "
                f"{s.avg_detection_time_ms:8.1f} |"
            )

        table.append("\n## Per-Image Comparison\n")
        table.append("| Image | OpenCV | ML | ML+Fallback | Winner |")
        table.append("|-------|--------|-------|-------------|--------|")

        # Per-image comparison
        image_names = sorted(all_results[list(all_results.keys())[0]][0].image_name
                            for r in all_results[list(all_results.keys())[0]])

        for idx, image_name in enumerate(sorted(self.ground_truth.keys())):
            row = [f"| {image_name:30s} |"]

            method_errors = {}
            for method_name in ['opencv', 'ml', 'ml_with_fallback']:
                if method_name not in all_results:
                    row.append(" N/A |")
                    continue

                result = all_results[method_name][idx]
                if result.success:
                    row.append(f" {result.avg_error:6.2f}px |")
                    method_errors[method_name] = result.avg_error
                else:
                    row.append(" FAIL |")
                    method_errors[method_name] = float('inf')

            # Determine winner
            if method_errors:
                winner = min(method_errors.items(), key=lambda x: x[1])
                if winner[1] != float('inf'):
                    row.append(f" {winner[0]:11s} |")
                else:
                    row.append(" none |")
            else:
                row.append(" - |")

            table.append("".join(row))

        return "\n".join(table)

    def save_results(
        self,
        all_results: Dict[str, List[DetectionResult]],
        comparison_table: str
    ):
        """
        Save all results to files.

        Args:
            all_results: Results for all methods
            comparison_table: Markdown comparison table
        """
        # Save markdown table
        table_path = self.output_dir / "comparison_table.md"
        with open(table_path, 'w') as f:
            f.write(comparison_table)
        print(f"✓ Comparison table saved to: {table_path}")

        # Save JSON results
        json_results = {}
        for method_name, results in all_results.items():
            json_results[method_name] = []
            for result in results:
                result_dict = asdict(result)
                # Convert numpy arrays to lists for JSON serialization
                if result_dict['corners'] is not None:
                    result_dict['corners'] = result_dict['corners'].tolist()
                json_results[method_name].append(result_dict)

        json_path = self.output_dir / "comparison_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"✓ JSON results saved to: {json_path}")

        # Compute and save statistics
        stats = {}
        for method_name, results in all_results.items():
            stats[method_name] = asdict(
                self.compute_statistics(results, method_name)
            )

        stats_path = self.output_dir / "comparison_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Statistics saved to: {stats_path}")

    def run_comparison(self):
        """Run full comparison and generate reports"""
        print("\n" + "="*70)
        print("PAPER DETECTION METHOD COMPARISON")
        print("="*70)
        print(f"\nTest images: {len(self.ground_truth)}")
        print(f"Detection methods: {', '.join(self.detectors.keys())}")
        print(f"Output directory: {self.output_dir}\n")

        # Evaluate all methods
        all_results = self.evaluate_all_methods()

        # Generate comparison table
        comparison_table = self.generate_comparison_table(all_results)

        # Print table to console
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70 + "\n")
        print(comparison_table)

        # Save results
        self.save_results(all_results, comparison_table)

        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare paper detection methods (OpenCV vs ML)'
    )
    parser.add_argument(
        '--images-dir',
        type=Path,
        default=None,
        help='Directory containing test images'
    )
    parser.add_argument(
        '--ground-truth',
        type=Path,
        default=None,
        help='Path to ground truth JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--ml-model-path',
        type=str,
        default=None,
        help='Path to ML model file (optional)'
    )

    args = parser.parse_args()

    # Use defaults if not specified
    script_dir = Path(__file__).parent
    images_dir = args.images_dir or (script_dir / "test_images")
    ground_truth = args.ground_truth or (script_dir / "test_corners_ground_truth.json")
    output_dir = args.output_dir or (script_dir / "output" / "comparison")

    # Validate paths
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1

    if not ground_truth.exists():
        print(f"Error: Ground truth file not found: {ground_truth}")
        return 1

    # Run comparison
    comparator = DetectionMethodComparator(
        images_dir=images_dir,
        ground_truth_path=ground_truth,
        output_dir=output_dir,
        ml_model_path=args.ml_model_path
    )

    comparator.run_comparison()

    return 0


if __name__ == "__main__":
    sys.exit(main())
