"""
Visualizer
===========
Draws annotated bounding boxes, violation labels, confidence bars,
and plate text on images/video frames. Color-coded by violation type.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


# Color palette for violation types (BGR format)
VIOLATION_COLORS = {
    "NO_HELMET": (0, 0, 255),           # Red
    "TRIPLE_RIDING": (0, 100, 255),     # Orange
    "RED_LIGHT_VIOLATION": (0, 0, 200), # Dark Red
    "ILLEGAL_PARKING": (255, 0, 255),   # Magenta
    "STOP_LINE_VIOLATION": (0, 165, 255),  # Orange-Yellow
    "MODIFIED_VEHICLE": (128, 0, 128),  # Purple
}

# Colors for detection types
DETECTION_COLORS = {
    "vehicle": (255, 200, 0),      # Cyan-ish
    "rider": (0, 255, 0),          # Green
    "traffic_light": (0, 255, 255),  # Yellow
    "stop_line": (255, 255, 0),    # Cyan
    "plate": (255, 128, 0),        # Blue-Orange
}

SEVERITY_COLORS = {
    "HIGH": (0, 0, 255),
    "MEDIUM": (0, 165, 255),
    "LOW": (0, 255, 255),
}


def draw_bbox(
    frame: np.ndarray,
    bbox: List[float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw a bounding box with a label on the frame."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_confidence_bar(
    frame: np.ndarray,
    bbox: List[float],
    confidence: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Draw a small confidence bar below the bounding box."""
    x1, y2 = int(bbox[0]), int(bbox[3])
    bar_width = int((bbox[2] - bbox[0]))
    bar_height = 6

    # Background bar (gray)
    cv2.rectangle(
        frame, (x1, y2 + 2), (x1 + bar_width, y2 + 2 + bar_height),
        (100, 100, 100), -1,
    )
    # Filled portion
    fill_w = int(bar_width * confidence)
    cv2.rectangle(
        frame, (x1, y2 + 2), (x1 + fill_w, y2 + 2 + bar_height),
        color, -1,
    )
    return frame


def annotate_frame(
    frame: np.ndarray,
    report_dict: Dict[str, Any],
    draw_detections: bool = True,
    draw_violations: bool = True,
    draw_plates: bool = True,
) -> np.ndarray:
    """
    Annotate a frame with all detections, violations, and plates.

    Args:
        frame: BGR numpy array (will be modified in-place).
        report_dict: Output from ViolationReport.to_dict().
        draw_detections: Draw raw YOLO detections.
        draw_violations: Draw violation overlays.
        draw_plates: Draw plate annotations.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # ── Draw raw detections (subtle) ──
    if draw_detections:
        for det in report_dict.get("all_detections", []):  # noqa: F841 (used below)
            pass  # Skip raw detections to keep frame clean, violations are enough

    # ── Draw violations (prominent) ──
    if draw_violations:
        for violation in report_dict.get("violations", []):
            v_type = violation["type"]
            confidence = violation["confidence"]
            severity = violation["severity"]
            bbox = violation["bbox"]

            color = VIOLATION_COLORS.get(v_type, (0, 0, 255))

            # Main label
            label = f"{v_type} {confidence:.0%} [{severity}]"
            draw_bbox(annotated, bbox, label, color, thickness=3)
            draw_confidence_bar(annotated, bbox, confidence, color)

            # Draw plate text if attached
            if draw_plates and violation.get("plate"):
                plate = violation["plate"]
                plate_label = f"PLATE: {plate['text']} ({plate['confidence']:.0%})"
                plate_bbox = plate.get("bbox")
                if plate_bbox:
                    draw_bbox(annotated, plate_bbox, plate_label, (255, 128, 0), thickness=2)

    # ── Draw standalone plates (not linked to violations) ──
    if draw_plates:
        for plate in report_dict.get("plates", []):
            plate_bbox = plate.get("bbox")
            plate_text = plate.get("plate_text", "")
            plate_conf = plate.get("combined_confidence", 0)
            if plate_bbox:
                label = f"{plate_text} ({plate_conf:.0%})"
                draw_bbox(annotated, plate_bbox, label, (255, 128, 0), thickness=2)

    # ── Info overlay (top-left) ──
    info_lines = [
        f"Frame: {report_dict.get('frame_id', '?')}",
        f"Violations: {report_dict.get('violation_count', 0)}",
        f"Plates: {report_dict.get('plate_count', 0)}",
        f"Time: {report_dict.get('processing_time_ms', 0):.0f}ms",
    ]

    y_offset = 25
    for line in info_lines:
        cv2.putText(
            annotated, line, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        y_offset += 22

    return annotated


def annotate_video(
    input_path: str,
    output_path: str,
    reports: List[Dict[str, Any]],
    fps: float = None,
):
    """
    Create an annotated video from original video + reports.

    Args:
        input_path: Original video path.
        output_path: Output annotated video path.
        reports: List of report dicts keyed by frame_id.
        fps: Output FPS (auto-detected if None).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Index reports by frame_id
    report_map = {r["frame_id"]: r for r in reports}
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in report_map:
            frame = annotate_frame(frame, report_map[frame_id])

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"[Visualizer] Annotated video saved to {output_path}")
