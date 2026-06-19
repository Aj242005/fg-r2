"""
Red Light Violation Analyzer
==============================
Detects vehicles crossing a stop line while the traffic light is red.
Uses spatial analysis to determine if vehicle bottom crosses below the stop line.
"""

from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.violation_detector import Detection
from engine.spatial_utils import bbox_bottom_center, bbox_center


class RedLightViolationAnalyzer:
    """Analyze detections for red light jumping violations."""

    VIOLATION_TYPE = "RED_LIGHT_VIOLATION"

    def analyze(
        self,
        detections: List[Detection],
        img_width: int = 1920,
        img_height: int = 1080,
        tracked_objects: dict = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect red light violations.

        Logic:
            1. Check if a red light is detected in frame
            2. Find stop line position
            3. For each vehicle, check if its bottom edge is past the stop line
            4. If tracked_objects provided, also check if vehicle *crossed* the line

        Returns:
            List of violation dicts.
        """
        violations = []

        # Find red light detections
        red_lights = [
            d for d in detections if d.class_id == config.RED_LIGHT_CLASS
        ]

        if not red_lights:
            return violations  # No red light → no violation

        # Find stop lines
        stop_lines = [
            d for d in detections if d.class_id == config.STOP_LINE_CLASS
        ]

        # Best red light (highest confidence)
        red_light = max(red_lights, key=lambda d: d.confidence)

        # Get stop line Y position (bottom of stop line bbox)
        if stop_lines:
            stop_line = max(stop_lines, key=lambda d: d.confidence)
            # Stop line Y = center Y of the stop line bbox
            stop_line_y = bbox_center(stop_line.bbox)[1]
            stop_line_conf = stop_line.confidence
        else:
            # If no stop line detected, estimate: red light usually at top,
            # stop line ~60-70% down the frame
            stop_line_y = img_height * 0.65
            stop_line_conf = 0.3  # low confidence since estimated

        # Check each vehicle
        vehicles = [
            d for d in detections if d.class_id in config.VEHICLE_CLASSES
        ]

        for vehicle in vehicles:
            vehicle_bottom = bbox_bottom_center(vehicle.bbox)

            # Is the vehicle past the stop line?
            is_past_line = vehicle_bottom[1] > stop_line_y

            if not is_past_line:
                continue

            # How far past the line (normalized)
            overshoot = (vehicle_bottom[1] - stop_line_y) / img_height
            crossing_certainty = min(1.0, overshoot * 5)  # scale up

            # Check temporal crossing if tracker available
            temporal_conf = 0.5  # neutral default
            if tracked_objects:
                for obj_id, tracked in tracked_objects.items():
                    if tracked.class_id == vehicle.class_id:
                        prev = tracked.prev_centroid
                        curr = tracked.centroid
                        if prev and prev[1] <= stop_line_y < curr[1]:
                            temporal_conf = 0.95  # definite crossing observed
                            break

            # Composite confidence
            composite = (
                0.30 * red_light.confidence
                + 0.25 * vehicle.confidence
                + 0.20 * stop_line_conf
                + 0.15 * crossing_certainty
                + 0.10 * temporal_conf
            )

            if composite >= config.RED_LIGHT_VIOLATION_THRESHOLD * 0.5:
                vehicle_type = "two_wheeler"
                if vehicle.class_id in config.THREE_WHEELER_CLASSES:
                    vehicle_type = "three_wheeler"
                elif vehicle.class_id in config.FOUR_WHEELER_CLASSES:
                    vehicle_type = "four_wheeler"

                violations.append(
                    {
                        "type": self.VIOLATION_TYPE,
                        "confidence": round(composite, 4),
                        "severity": config.get_severity(composite),
                        "bbox": [round(v, 2) for v in vehicle.bbox],
                        "details": {
                            "vehicle_class": vehicle.class_name,
                            "vehicle_type": vehicle_type,
                            "vehicle_confidence": round(vehicle.confidence, 4),
                            "red_light_confidence": round(red_light.confidence, 4),
                            "stop_line_confidence": round(stop_line_conf, 4),
                            "crossing_certainty": round(crossing_certainty, 4),
                            "temporal_confidence": round(temporal_conf, 4),
                            "stop_line_y": round(stop_line_y, 2),
                            "vehicle_bottom_y": round(vehicle_bottom[1], 2),
                        },
                    }
                )

        return violations
