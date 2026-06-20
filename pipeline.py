"""
Dual-Model Parallel Pipeline
==============================
Orchestrates both models (violation detection + plate OCR) in parallel.
Merges results by spatial correlation.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import config
from engine.violation_detector import ViolationDetector, Detection
from engine.plate_reader import PlateReader, PlateResult
from engine.tracker import CentroidTracker
from engine.spatial_utils import compute_iou

from violations import (
    HelmetViolationAnalyzer,
    TriplingViolationAnalyzer,
    RedLightViolationAnalyzer,
    IllegalParkingAnalyzer,
    StopLineViolationAnalyzer,
    VehicleModsAnalyzer,
)


class ViolationReport:
    """Structured report for a single frame analysis."""

    def __init__(
        self,
        frame_id: int,
        timestamp: float,
        violations: List[Dict[str, Any]],
        plates: List[Dict[str, Any]],
        all_detections: List[Dict[str, Any]],
        processing_time_ms: float,
    ):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.violations = violations
        self.plates = plates
        self.all_detections = all_detections
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "violation_count": len(self.violations),
            "plate_count": len(self.plates),
            "violations": self.violations,
            "plates": self.plates,
            "detections": self.all_detections,
            "detection_count": len(self.all_detections),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


class ViolationPipeline:
    """
    Main pipeline: runs both models in parallel and merges results.

    Usage:
        pipeline = ViolationPipeline()
        report = pipeline.process_frame(frame)
        # or
        reports = pipeline.process_video("video.mp4")
    """

    def __init__(
        self,
        violation_model_path: str = None,
        plate_model_path: str = None,
        device: str = "auto",
        enable_plate_reader: bool = True,
    ):
        """
        Initialize dual-model pipeline.

        Args:
            violation_model_path: Path to violation detection model (.pt).
            plate_model_path: Path to plate detection model (.pt).
            device: Device for inference.
            enable_plate_reader: If False, skip plate reading (if model unavailable).
        """
        # Load Model 1: Violation Detector
        print("=" * 60)
        print("INITIALIZING TRAFFIC VIOLATION PIPELINE")
        print("=" * 60)

        print("\n[Pipeline] Loading Model 1: Violation Detector...")
        self.violation_detector = ViolationDetector(
            model_path=violation_model_path, device=device
        )

        # Load Model 2: Plate Reader (optional, graceful degradation)
        self.plate_reader = None
        if enable_plate_reader:
            try:
                print("[Pipeline] Loading Model 2: Plate Reader...")
                self.plate_reader = PlateReader(
                    model_path=plate_model_path, device=device
                )
            except FileNotFoundError as e:
                print(f"[Pipeline] WARNING: Plate reader disabled — {e}")
                print("[Pipeline] System will work without plate reading.")

        # Initialize violation analyzers
        self.analyzers = [
            HelmetViolationAnalyzer(),
            TriplingViolationAnalyzer(),
            RedLightViolationAnalyzer(),
            IllegalParkingAnalyzer(),
            StopLineViolationAnalyzer(),
            VehicleModsAnalyzer(),
        ]

        # Object tracker for video mode
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)

        # Thread pool for parallel inference
        self.executor = ThreadPoolExecutor(max_workers=config.MODEL_WORKERS)

        self.frame_count = 0
        print("\n[Pipeline] ✅ Pipeline ready!")
        print(f"[Pipeline] Violation analyzers: {len(self.analyzers)}")
        print(f"[Pipeline] Plate reader: {'enabled' if self.plate_reader else 'disabled'}")
        print("=" * 60)

    def _run_violation_detection(self, frame: np.ndarray) -> List[Detection]:
        """Run Model 1 inference."""
        return self.violation_detector.detect(frame)

    def _run_plate_reading(self, frame: np.ndarray) -> List[PlateResult]:
        """Run Model 2 inference."""
        if self.plate_reader:
            return self.plate_reader.read_plates(frame)
        return []

    def _merge_plates_with_violations(
        self,
        violations: List[Dict[str, Any]],
        plates: List[PlateResult],
    ) -> List[Dict[str, Any]]:
        """
        Associate detected plates with violations by spatial overlap.
        If a plate bbox overlaps with a violation's vehicle bbox,
        attach the plate number to that violation.
        """
        for violation in violations:
            v_bbox = violation.get("bbox")
            if not v_bbox:
                continue

            best_plate = None
            best_iou = 0.0

            for plate in plates:
                iou = compute_iou(v_bbox, plate.bbox)
                # Also check if plate is inside the vehicle bbox
                if iou > best_iou or (iou == 0 and self._plate_inside_vehicle(plate.bbox, v_bbox)):
                    best_iou = iou
                    best_plate = plate

            if best_plate and (best_iou > 0.01 or self._plate_inside_vehicle(best_plate.bbox, v_bbox)):
                violation["plate"] = {
                    "text": best_plate.plate_text,
                    "confidence": round(best_plate.confidence, 4),
                    "bbox": [round(v, 2) for v in best_plate.bbox],
                }
            else:
                violation["plate"] = None

        return violations

    @staticmethod
    def _plate_inside_vehicle(plate_bbox, vehicle_bbox) -> bool:
        """Check if plate center is inside vehicle bbox."""
        pcx = (plate_bbox[0] + plate_bbox[2]) / 2
        pcy = (plate_bbox[1] + plate_bbox[3]) / 2
        return (
            vehicle_bbox[0] <= pcx <= vehicle_bbox[2]
            and vehicle_bbox[1] <= pcy <= vehicle_bbox[3]
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = None,
    ) -> ViolationReport:
        """
        Process a single frame through both models in parallel.

        Args:
            frame: BGR numpy array.
            frame_id: Optional frame identifier.

        Returns:
            ViolationReport with all violations, plates, and detections.
        """
        start_time = time.time()
        self.frame_count += 1
        fid = frame_id or self.frame_count

        h, w = frame.shape[:2]

        # ── PARALLEL INFERENCE ──
        future_violations = self.executor.submit(self._run_violation_detection, frame)
        future_plates = self.executor.submit(self._run_plate_reading, frame)

        # Wait for both to complete
        detections = future_violations.result()
        plates = future_plates.result()

        # ── UPDATE TRACKER ──
        tracker_input = [
            {"bbox": d.bbox, "class_id": d.class_id, "confidence": d.confidence}
            for d in detections
            if d.class_id in config.VEHICLE_CLASSES
        ]
        tracked_objects = self.tracker.update(tracker_input)

        # ── RUN ALL VIOLATION ANALYZERS ──
        all_violations = []
        for analyzer in self.analyzers:
            try:
                results = analyzer.analyze(
                    detections,
                    img_width=w,
                    img_height=h,
                    tracked_objects=tracked_objects,  # type: ignore
                )
                all_violations.extend(results)
            except TypeError:
                # Some analyzers don't accept tracked_objects
                results = analyzer.analyze(detections, img_width=w, img_height=h)
                all_violations.extend(results)

        # ── MERGE PLATES WITH VIOLATIONS ──
        if plates:
            all_violations = self._merge_plates_with_violations(all_violations, plates)

        # ── BUILD REPORT ──
        processing_time = (time.time() - start_time) * 1000

        report = ViolationReport(
            frame_id=fid,
            timestamp=time.time(),
            violations=all_violations,
            plates=[p.to_dict() for p in plates],
            all_detections=[d.to_dict() for d in detections],
            processing_time_ms=processing_time,
        )

        return report

    def process_image(self, image_path: str) -> ViolationReport:
        """Process a single image file."""
        import cv2

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.process_frame(frame, frame_id=0)

    def process_video(
        self,
        video_path: str,
        stride: int = None,
        max_frames: int = None,
        callback=None,
    ) -> List[ViolationReport]:
        """
        Process a video file frame by frame.

        Args:
            video_path: Path to video file.
            stride: Process every Nth frame (default from config).
            max_frames: Stop after N frames.
            callback: Optional function called with (frame_id, report) per frame.

        Returns:
            List of ViolationReports (only frames with violations).
        """
        import cv2

        stride = stride or config.VIDEO_STRIDE
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Pipeline] Video: {video_path}")
        print(f"[Pipeline] FPS: {fps:.1f}, Total frames: {total_frames}")
        print(f"[Pipeline] Processing every {stride} frame(s)")

        self.tracker.reset()
        reports = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_id >= max_frames:
                break

            if frame_id % stride == 0:
                report = self.process_frame(frame, frame_id=frame_id)

                if report.has_violations:
                    reports.append(report)

                if callback:
                    callback(frame_id, report)

            frame_id += 1

        cap.release()
        print(f"[Pipeline] Processed {frame_id} frames, {len(reports)} with violations")
        return reports

    def shutdown(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
        print("[Pipeline] Shutdown complete.")
