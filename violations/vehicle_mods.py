"""
Vehicle Modifications Analyzer
================================
Detects modified/illegal vehicles and overloading.
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.violation_detector import Detection


class VehicleModsAnalyzer:
    """Analyze detections for modified/illegal vehicle violations."""

    VIOLATION_TYPE = "MODIFIED_VEHICLE"

    def analyze(
        self,
        detections: List[Detection],
        img_width: int = 1920,
        img_height: int = 1080,
        tracked_objects: dict = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect modified/illegal vehicles.

        Logic:
            1. Find 'modified' class detections
            2. Associate with nearest vehicle
            3. Flag with confidence

        Returns:
            List of violation dicts.
        """
        violations = []

        modified_dets = [
            d for d in detections if d.class_id == config.MODIFIED_CLASS
        ]

        for mod_det in modified_dets:
            composite = mod_det.confidence

            if composite >= config.MODIFIED_VEHICLE_THRESHOLD:
                violations.append(
                    {
                        "type": self.VIOLATION_TYPE,
                        "confidence": round(composite, 4),
                        "severity": config.get_severity(composite),
                        "bbox": [round(v, 2) for v in mod_det.bbox],
                        "details": {
                            "detection_class": mod_det.class_name,
                            "detection_confidence": round(mod_det.confidence, 4),
                        },
                    }
                )

        return violations
