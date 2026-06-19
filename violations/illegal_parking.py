"""
Illegal Parking Analyzer
=========================
Detects vehicles parked in defined no-parking zones.
Uses temporal tracking to confirm stationarity.
"""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.violation_detector import Detection
from engine.spatial_utils import bbox_in_zone, bbox_center
from engine.tracker import TrackedObject


class IllegalParkingAnalyzer:
    """Analyze for illegal parking violations using zones + tracking."""

    VIOLATION_TYPE = "ILLEGAL_PARKING"

    def __init__(self, no_parking_zones: list = None):
        """
        Args:
            no_parking_zones: List of polygon zones (normalized 0-1 coordinates).
                              Falls back to config.NO_PARKING_ZONES.
        """
        self.zones = no_parking_zones or config.NO_PARKING_ZONES

    def analyze(
        self,
        detections: List[Detection],
        img_width: int = 1920,
        img_height: int = 1080,
        tracked_objects: dict = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect illegal parking.

        Logic:
            1. Check if any vehicles are within defined no-parking zones
            2. If tracked_objects available, check stationarity duration
            3. A vehicle is "parked" if it's been stationary for N frames

        For image-only mode (no tracking), flags vehicles in no-parking zones
        with lower confidence.

        Returns:
            List of violation dicts.
        """
        violations = []

        if not self.zones:
            return violations  # No zones defined, skip

        vehicles = [
            d for d in detections if d.class_id in config.VEHICLE_CLASSES
        ]

        for vehicle in vehicles:
            # Check if vehicle center is in any no-parking zone
            in_zone = False
            zone_idx = -1

            for idx, zone in enumerate(self.zones):
                if bbox_in_zone(vehicle.bbox, zone, img_width, img_height):
                    in_zone = True
                    zone_idx = idx
                    break

            if not in_zone:
                continue

            # Check stationarity from tracker
            stationary_frames = 0
            is_confirmed_stationary = False

            if tracked_objects:
                for obj_id, tracked in tracked_objects.items():
                    # Match tracked object to this detection by proximity
                    tc = tracked.centroid
                    dc = bbox_center(vehicle.bbox)
                    dist = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5

                    if dist < 50 and tracked.class_id == vehicle.class_id:
                        stationary_frames = tracked.stationary_frames
                        is_confirmed_stationary = (
                            stationary_frames >= config.PARKING_STATIONARY_FRAMES
                        )
                        break

            # Confidence calculation
            if is_confirmed_stationary:
                # High confidence: vehicle detected + in zone + confirmed stationary
                stationarity_score = min(
                    1.0, stationary_frames / (config.PARKING_STATIONARY_FRAMES * 2)
                )
                composite = (
                    0.35 * vehicle.confidence
                    + 0.35 * stationarity_score
                    + 0.30 * 1.0  # zone match is binary
                )
            else:
                # Lower confidence: only spatial (image-only mode)
                composite = vehicle.confidence * 0.4

            vehicle_type = "two_wheeler"
            if vehicle.class_id in config.THREE_WHEELER_CLASSES:
                vehicle_type = "three_wheeler"
            elif vehicle.class_id in config.FOUR_WHEELER_CLASSES:
                vehicle_type = "four_wheeler"

            if composite >= config.PARKING_VIOLATION_THRESHOLD * 0.5:
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
                            "zone_index": zone_idx,
                            "stationary_frames": stationary_frames,
                            "is_confirmed_stationary": is_confirmed_stationary,
                        },
                    }
                )

        return violations
