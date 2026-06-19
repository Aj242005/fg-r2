"""
Stop Line Violation Analyzer
==============================
Detects vehicles crossing or past the stop line
(independent of traffic light state — yellow/green included).
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.violation_detector import Detection
from engine.spatial_utils import bbox_bottom_center, bbox_center


class StopLineViolationAnalyzer:
    """Analyze detections for stop line crossing violations."""

    VIOLATION_TYPE = "STOP_LINE_VIOLATION"

    def analyze(
        self,
        detections: List[Detection],
        img_width: int = 1920,
        img_height: int = 1080,
        tracked_objects: dict = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect vehicles crossing stop lines during non-green signals.

        Logic:
            1. Find stop line detections
            2. Check traffic light state (skip if green)
            3. Flag vehicles past the stop line

        Returns:
            List of violation dicts.
        """
        violations = []

        # Find stop lines
        stop_lines = [
            d for d in detections if d.class_id == config.STOP_LINE_CLASS
        ]

        if not stop_lines:
            return violations

        # Check if green light — if green, no stop line violation
        green_lights = [
            d for d in detections if d.class_id == config.GREEN_LIGHT_CLASS
        ]
        if green_lights:
            return violations  # Green light = go, no violation

        # Best stop line
        stop_line = max(stop_lines, key=lambda d: d.confidence)
        stop_line_y = bbox_center(stop_line.bbox)[1]

        # Check traffic light state
        red_lights = [d for d in detections if d.class_id == config.RED_LIGHT_CLASS]
        yellow_lights = [d for d in detections if d.class_id == config.YELLOW_LIGHT_CLASS]
        signal_state = "red" if red_lights else ("yellow" if yellow_lights else "unknown")

        vehicles = [
            d for d in detections if d.class_id in config.VEHICLE_CLASSES
        ]

        for vehicle in vehicles:
            vehicle_bottom = bbox_bottom_center(vehicle.bbox)

            if vehicle_bottom[1] <= stop_line_y:
                continue  # Not past the line

            overshoot = (vehicle_bottom[1] - stop_line_y) / img_height
            crossing_certainty = min(1.0, overshoot * 5)

            composite = (
                0.40 * vehicle.confidence
                + 0.35 * stop_line.confidence
                + 0.25 * crossing_certainty
            )

            if composite >= config.STOP_LINE_VIOLATION_THRESHOLD * 0.5:
                violations.append(
                    {
                        "type": self.VIOLATION_TYPE,
                        "confidence": round(composite, 4),
                        "severity": config.get_severity(composite),
                        "bbox": [round(v, 2) for v in vehicle.bbox],
                        "details": {
                            "vehicle_class": vehicle.class_name,
                            "vehicle_confidence": round(vehicle.confidence, 4),
                            "stop_line_confidence": round(stop_line.confidence, 4),
                            "crossing_certainty": round(crossing_certainty, 4),
                            "signal_state": signal_state,
                        },
                    }
                )

        return violations
