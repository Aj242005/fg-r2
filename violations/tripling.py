"""
Tripling Violation Analyzer
=============================
Detects triple riding (3+ persons on a two-wheeler).
Counts rider/helmet detections that overlap with a motorcycle bounding box.
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.violation_detector import Detection
from engine.spatial_utils import compute_iou, bbox_overlap_ratio, expand_bbox


class TriplingViolationAnalyzer:
    """Analyze detections for triple riding violations."""

    VIOLATION_TYPE = "TRIPLE_RIDING"

    def analyze(
        self,
        detections: List[Detection],
        img_width: int = 1920,
        img_height: int = 1080,
    ) -> List[Dict[str, Any]]:
        """
        Detect triple riding by counting riders associated with each two-wheeler.

        Logic:
            1. Find all two-wheeler detections
            2. For each two-wheeler, expand bbox slightly
            3. Count rider-class detections (helmet/no-helmet) overlapping
            4. If count >= 3, flag as tripling

        Returns:
            List of violation dicts.
        """
        violations = []

        # Get two-wheelers
        two_wheelers = [
            d for d in detections if d.class_id in config.TWO_WHEELER_CLASSES
        ]

        # Get all rider-related detections
        riders = [d for d in detections if d.class_id in config.RIDER_CLASSES]

        for tw in two_wheelers:
            # Expand vehicle bbox to catch riders slightly outside strict bbox
            expanded = expand_bbox(tw.bbox, factor=0.15)
            associated_riders = []

            for rider in riders:
                overlap = bbox_overlap_ratio(rider.bbox, expanded)
                iou = compute_iou(rider.bbox, expanded)

                # Consider associated if significant overlap
                if overlap > 0.2 or iou > 0.1:
                    associated_riders.append(
                        {
                            "detection": rider,
                            "overlap": overlap,
                            "iou": iou,
                        }
                    )

            rider_count = len(associated_riders)

            if rider_count >= config.TRIPLING_MIN_RIDERS:
                # Composite confidence
                rider_confs = [r["detection"].confidence for r in associated_riders]
                avg_rider_conf = sum(rider_confs) / len(rider_confs)
                avg_overlap = sum(r["overlap"] for r in associated_riders) / len(
                    associated_riders
                )

                composite = (
                    0.40 * tw.confidence
                    + 0.40 * avg_rider_conf
                    + 0.20 * min(1.0, avg_overlap)
                )

                violations.append(
                    {
                        "type": self.VIOLATION_TYPE,
                        "confidence": round(composite, 4),
                        "severity": config.get_severity(composite),
                        "bbox": [round(v, 2) for v in tw.bbox],
                        "details": {
                            "vehicle_class": tw.class_name,
                            "vehicle_confidence": round(tw.confidence, 4),
                            "rider_count": rider_count,
                            "rider_confidences": [round(c, 4) for c in rider_confs],
                            "avg_overlap": round(avg_overlap, 4),
                        },
                    }
                )

        return violations
