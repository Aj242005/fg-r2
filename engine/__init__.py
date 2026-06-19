"""Engine package — core detection, OCR, tracking, and spatial utilities."""

from .violation_detector import ViolationDetector
from .plate_reader import PlateReader
from .tracker import CentroidTracker
from .spatial_utils import (
    compute_iou,
    bbox_center,
    bbox_distance,
    point_in_polygon,
    crosses_line,
    bbox_contains_point,
    bbox_overlap_ratio,
)

__all__ = [
    "ViolationDetector",
    "PlateReader",
    "CentroidTracker",
    "compute_iou",
    "bbox_center",
    "bbox_distance",
    "point_in_polygon",
    "crosses_line",
    "bbox_contains_point",
    "bbox_overlap_ratio",
]
