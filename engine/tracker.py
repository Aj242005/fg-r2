"""
Centroid Tracker
=================
Simple, lightweight multi-object tracker using centroid distance matching.
Tracks objects across video frames to support temporal analysis
(e.g., stationary detection for parking, line-crossing for red light).
"""

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist


class TrackedObject:
    """Represents a tracked object across frames."""

    def __init__(self, object_id: int, centroid: Tuple[float, float], bbox: List[float], class_id: int):
        self.object_id = object_id
        self.centroid = centroid
        self.bbox = bbox
        self.class_id = class_id
        self.positions: List[Tuple[float, float]] = [centroid]
        self.frames_since_seen = 0
        self.total_frames = 1

    @property
    def is_stationary(self) -> bool:
        """Check if object has barely moved over its tracked lifetime."""
        if len(self.positions) < 10:
            return False
        recent = self.positions[-10:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        return spread < 15  # pixels

    @property
    def stationary_frames(self) -> int:
        """Count consecutive frames the object hasn't moved significantly."""
        count = 0
        for i in range(len(self.positions) - 1, 0, -1):
            dx = abs(self.positions[i][0] - self.positions[i - 1][0])
            dy = abs(self.positions[i][1] - self.positions[i - 1][1])
            if dx < 5 and dy < 5:
                count += 1
            else:
                break
        return count

    @property
    def prev_centroid(self) -> Optional[Tuple[float, float]]:
        """Return the previous centroid if available."""
        if len(self.positions) >= 2:
            return self.positions[-2]
        return None


class CentroidTracker:
    """
    Track objects across frames using centroid distance matching.

    Args:
        max_disappeared: Number of frames an object can be absent before removal.
        max_distance: Maximum centroid distance for matching (pixels).
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self.next_id = 0
        self.objects: OrderedDict[int, TrackedObject] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[float, float], bbox: List[float], class_id: int) -> int:
        """Register a new object and return its ID."""
        obj = TrackedObject(self.next_id, centroid, bbox, class_id)
        self.objects[self.next_id] = obj
        self.next_id += 1
        return obj.object_id

    def deregister(self, object_id: int):
        """Remove a tracked object."""
        del self.objects[object_id]

    def update(
        self, detections: List[dict]
    ) -> Dict[int, TrackedObject]:
        """
        Update tracked objects with new frame detections.

        Args:
            detections: List of dicts with keys 'bbox' [x1,y1,x2,y2],
                        'class_id' (int), 'confidence' (float).

        Returns:
            Dictionary of object_id -> TrackedObject for all currently tracked objects.
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].frames_since_seen += 1
                if self.objects[obj_id].frames_since_seen > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # Compute centroids for incoming detections
        input_centroids = []
        input_bboxes = []
        input_classes = []
        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)
            input_classes.append(det.get("class_id", -1))

        input_centroids = np.array(input_centroids)

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(
                    tuple(input_centroids[i]),
                    input_bboxes[i],
                    input_classes[i],
                )
            return self.objects

        # Match existing objects to new detections via centroid distance
        object_ids = list(self.objects.keys())
        existing_centroids = np.array(
            [self.objects[oid].centroid for oid in object_ids]
        )

        # Distance matrix: existing x input
        dist_matrix = cdist(existing_centroids, input_centroids)

        # Greedy matching: sort by distance, match closest pairs
        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if dist_matrix[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            new_centroid = tuple(input_centroids[col])

            self.objects[obj_id].centroid = new_centroid
            self.objects[obj_id].bbox = input_bboxes[col]
            self.objects[obj_id].class_id = input_classes[col]
            self.objects[obj_id].positions.append(new_centroid)
            self.objects[obj_id].frames_since_seen = 0
            self.objects[obj_id].total_frames += 1

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (disappeared)
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.objects[obj_id].frames_since_seen += 1
            if self.objects[obj_id].frames_since_seen > self.max_disappeared:
                self.deregister(obj_id)

        # Handle unmatched new detections (new objects)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.register(
                tuple(input_centroids[col]),
                input_bboxes[col],
                input_classes[col],
            )

        return self.objects

    def reset(self):
        """Clear all tracked objects."""
        self.objects.clear()
        self.next_id = 0
