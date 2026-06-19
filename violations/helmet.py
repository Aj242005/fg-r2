"""
Helmet Violation Analyzer
==========================
Detects riders without helmets on 2-wheelers and 3-wheelers.
Associates helmet/no-helmet detections with nearby vehicle detections.
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.violation_detector import Detection
from engine.spatial_utils import compute_iou, bbox_distance, bbox_overlap_ratio


class HelmetViolationAnalyzer:
    """Analyze detections for helmet violations."""

    VIOLATION_TYPE = "NO_HELMET"

    def analyze(
        self,
        detections: List[Detection],
        img_width: int = 1920,
        img_height: int = 1080,
    ) -> List[Dict[str, Any]]:
        """
        Find helmet violations by associating no-helmet riders with vehicles.

        Logic:
            1. Find all 'without_helmet' / 'rider_without_helmet' detections
            2. Find all two-wheeler and three-wheeler detections
            3. Associate each no-helmet detection with the nearest vehicle
            4. Compute composite confidence

        Returns:
            List of violation dicts with type, confidence, severity, bbox, details.
        """
        violations = []

        # Get no-helmet detections
        no_helmet_dets = [
            d for d in detections if d.class_id in config.HELMET_ABSENT_CLASSES
        ]

        # Get vehicle detections (2-wheelers and 3-wheelers)
        vehicle_dets = [
            d
            for d in detections
            if d.class_id in config.TWO_WHEELER_CLASSES | config.THREE_WHEELER_CLASSES
        ]

        for helmet_det in no_helmet_dets:
            best_vehicle = None
            best_score = 0.0

            for vehicle_det in vehicle_dets:
                # Check spatial association: IoU or proximity
                iou = compute_iou(helmet_det.bbox, vehicle_det.bbox)
                overlap = bbox_overlap_ratio(helmet_det.bbox, vehicle_det.bbox)
                dist = bbox_distance(helmet_det.bbox, vehicle_det.bbox)

                # Normalize distance score (closer = higher)
                max_dist = max(img_width, img_height) * 0.3
                dist_score = max(0.0, 1.0 - dist / max_dist)

                # Combined spatial score
                spatial_score = max(iou, overlap, dist_score * 0.5)

                if spatial_score > best_score:
                    best_score = spatial_score
                    best_vehicle = vehicle_det

            # Compute composite confidence
            if best_vehicle:
                detection_conf = helmet_det.confidence
                vehicle_conf = best_vehicle.confidence
                spatial_conf = best_score

                composite = (
                    0.50 * detection_conf
                    + 0.25 * vehicle_conf
                    + 0.25 * spatial_conf
                )

                vehicle_type = (
                    "two_wheeler"
                    if best_vehicle.class_id in config.TWO_WHEELER_CLASSES
                    else "three_wheeler"
                )
            else:
                # No vehicle found, still flag but with lower confidence
                composite = helmet_det.confidence * 0.6
                vehicle_type = "unknown"

            if composite >= config.HELMET_VIOLATION_THRESHOLD * 0.5:
                violations.append(
                    {
                        "type": self.VIOLATION_TYPE,
                        "confidence": round(composite, 4),
                        "severity": config.get_severity(composite),
                        "bbox": [round(v, 2) for v in helmet_det.bbox],
                        "vehicle_bbox": (
                            [round(v, 2) for v in best_vehicle.bbox]
                            if best_vehicle
                            else None
                        ),
                        "details": {
                            "rider_class": helmet_det.class_name,
                            "rider_confidence": round(helmet_det.confidence, 4),
                            "vehicle_type": vehicle_type,
                            "vehicle_confidence": (
                                round(best_vehicle.confidence, 4)
                                if best_vehicle
                                else None
                            ),
                            "spatial_score": round(best_score, 4),
                        },
                    }
                )

        return violations
