"""
Spatial Utility Functions
==========================
IoU, proximity, point-in-polygon, line crossing detection,
bounding box overlap and containment checks.
"""

import numpy as np
from typing import Tuple, List


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def bbox_center(box: List[float]) -> Tuple[float, float]:
    """Return centroid (cx, cy) of a [x1, y1, x2, y2] bbox."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_distance(box_a: List[float], box_b: List[float]) -> float:
    """Euclidean distance between centroids of two bboxes."""
    ca = bbox_center(box_a)
    cb = bbox_center(box_b)
    return np.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def bbox_bottom_center(box: List[float]) -> Tuple[float, float]:
    """Bottom-center point of a bbox (useful for line-crossing checks)."""
    return ((box[0] + box[2]) / 2.0, box[3])


def bbox_contains_point(box: List[float], point: Tuple[float, float]) -> bool:
    """Check if a point falls inside a bounding box."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def bbox_overlap_ratio(inner: List[float], outer: List[float]) -> float:
    """
    What fraction of 'inner' bbox is contained within 'outer' bbox.
    Returns a ratio in [0, 1].
    """
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])

    return inter_area / inner_area if inner_area > 0 else 0.0


def point_in_polygon(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
) -> bool:
    """
    Ray-casting algorithm to check if a point is inside a polygon.
    Polygon is a list of (x, y) vertices.
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def crosses_line(
    prev_pos: Tuple[float, float],
    curr_pos: Tuple[float, float],
    line_y: float,
) -> bool:
    """
    Check if an object crossed a horizontal line (defined by Y coordinate)
    between two consecutive frames.
    Returns True if object moved from above to below the line.
    """
    return prev_pos[1] <= line_y < curr_pos[1]


def bbox_in_zone(
    bbox: List[float],
    zone: List[Tuple[float, float]],
    img_width: int,
    img_height: int,
) -> bool:
    """
    Check if the center of a bbox falls within a normalized polygon zone.
    Zone vertices are in [0, 1] range; they get scaled to image dimensions.
    """
    cx, cy = bbox_center(bbox)
    # Normalize bbox center to [0, 1]
    norm_cx = cx / img_width
    norm_cy = cy / img_height
    return point_in_polygon((norm_cx, norm_cy), zone)


def expand_bbox(bbox: List[float], factor: float = 0.1) -> List[float]:
    """Expand a bbox by a given factor on all sides."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    dx = w * factor
    dy = h * factor
    return [
        bbox[0] - dx,
        bbox[1] - dy,
        bbox[2] + dx,
        bbox[3] + dy,
    ]
