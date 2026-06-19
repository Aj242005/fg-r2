"""
Central Configuration for Traffic Violation Detection System
=============================================================
All model paths, thresholds, class mappings, and zone definitions.
Users only need to drop trained .pt files into the models/ folder.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"

# Model file paths — drop your trained .pt files here
VIOLATION_MODEL_PATH = MODELS_DIR / "violation_model.pt"
PLATE_MODEL_PATH = MODELS_DIR / "plate_model.pt"

# Fallback: check parent directory for existing best.pt
FALLBACK_VIOLATION_MODEL = BASE_DIR.parent / "best.pt"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# MODEL 1: VIOLATION DETECTION — CLASS MAPPING (21 classes)
# ============================================================================

VIOLATION_CLASSES = {
    0: "bus",
    1: "car",
    2: "motorcycle",
    3: "truck",
    4: "three_wheeler",
    5: "tractor",
    6: "van",
    7: "vikram",
    8: "two_wheeler",
    9: "bike",
    10: "with_helmet",
    11: "without_helmet",
    12: "rider_with_helmet",
    13: "rider_without_helmet",
    14: "red_light",
    15: "green_light",
    16: "yellow_light",
    17: "traffic_light",
    18: "stop_line",
    19: "fixed_obstacle",
    20: "modified",
}

# Semantic groupings
VEHICLE_CLASSES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
TWO_WHEELER_CLASSES = {2, 8, 9}           # motorcycle, two_wheeler, bike
THREE_WHEELER_CLASSES = {4, 7}             # three_wheeler, vikram
FOUR_WHEELER_CLASSES = {0, 1, 3, 6}        # bus, car, truck, van
HELMET_PRESENT_CLASSES = {10, 12}          # with_helmet, rider_with_helmet
HELMET_ABSENT_CLASSES = {11, 13}           # without_helmet, rider_without_helmet
RIDER_CLASSES = {10, 11, 12, 13}           # all helmet-related (implies rider)
TRAFFIC_LIGHT_CLASSES = {14, 15, 16, 17}   # all signal states
RED_LIGHT_CLASS = 14
GREEN_LIGHT_CLASS = 15
YELLOW_LIGHT_CLASS = 16
STOP_LINE_CLASS = 18
MODIFIED_CLASS = 20

# ============================================================================
# MODEL 2: PLATE DETECTION — CLASS MAPPING
# ============================================================================

PLATE_CLASSES = {
    0: "license_plate",
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

# Detection thresholds
VIOLATION_CONF_THRESHOLD = 0.25      # YOLO detection confidence
PLATE_CONF_THRESHOLD = 0.30          # Plate detection confidence
OCR_CONF_THRESHOLD = 0.40            # OCR text confidence

# Violation-specific thresholds
HELMET_VIOLATION_THRESHOLD = 0.50
TRIPLING_VIOLATION_THRESHOLD = 0.45
RED_LIGHT_VIOLATION_THRESHOLD = 0.55
PARKING_VIOLATION_THRESHOLD = 0.50
STOP_LINE_VIOLATION_THRESHOLD = 0.50
MODIFIED_VEHICLE_THRESHOLD = 0.40

# NMS IOU threshold
NMS_IOU_THRESHOLD = 0.45

# ============================================================================
# SEVERITY LEVELS
# ============================================================================

SEVERITY_HIGH = "HIGH"       # confidence >= 0.80
SEVERITY_MEDIUM = "MEDIUM"   # 0.50 <= confidence < 0.80
SEVERITY_LOW = "LOW"         # confidence < 0.50

def get_severity(confidence: float) -> str:
    """Map confidence to severity level."""
    if confidence >= 0.80:
        return SEVERITY_HIGH
    elif confidence >= 0.50:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW

# ============================================================================
# SPATIAL ANALYSIS
# ============================================================================

# Proximity threshold for associating riders with vehicles (fraction of image)
RIDER_VEHICLE_IOU_THRESHOLD = 0.15
RIDER_VEHICLE_PROXIMITY_PX = 80  # pixels

# For tripling: minimum riders on a 2-wheeler to flag
TRIPLING_MIN_RIDERS = 3

# For parking: minimum stationary frames to flag
PARKING_STATIONARY_FRAMES = 30  # ~1 second at 30fps
PARKING_MOVEMENT_THRESHOLD = 10  # pixels of centroid movement

# ============================================================================
# NO-PARKING ZONES (defined as polygon vertices, normalized 0-1)
# Users can add/modify zones for their specific camera setup
# ============================================================================

NO_PARKING_ZONES = [
    # Example zone 1: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    # [(0.0, 0.7), (0.3, 0.7), (0.3, 1.0), (0.0, 1.0)],
]

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

INFERENCE_IMG_SIZE = 640
MAX_DETECTIONS = 100
DEVICE = "auto"  # "auto", "cpu", "0" (GPU 0), "cuda"

# Video processing
VIDEO_BATCH_SIZE = 1
VIDEO_STRIDE = 1  # process every Nth frame (1 = all frames)

# ============================================================================
# FASTAPI SERVER
# ============================================================================

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
MAX_UPLOAD_SIZE_MB = 500
STREAM_RECONNECT_DELAY = 5  # seconds

# Thread pool for parallel model inference
MODEL_WORKERS = 2  # one per model

# ============================================================================
# HELPER: Resolve model path with fallback
# ============================================================================

def get_violation_model_path() -> Path:
    """Return the violation model path, falling back to parent best.pt."""
    if VIOLATION_MODEL_PATH.exists():
        return VIOLATION_MODEL_PATH
    if FALLBACK_VIOLATION_MODEL.exists():
        return FALLBACK_VIOLATION_MODEL
    raise FileNotFoundError(
        f"No violation model found. Place 'violation_model.pt' in {MODELS_DIR} "
        f"or ensure 'best.pt' exists at {FALLBACK_VIOLATION_MODEL}"
    )

def get_plate_model_path() -> Path:
    """Return the plate model path."""
    if PLATE_MODEL_PATH.exists():
        return PLATE_MODEL_PATH
    raise FileNotFoundError(
        f"No plate model found. Place 'plate_model.pt' in {MODELS_DIR}. "
        f"Train one using training/train_plate_ocr_model.py on Kaggle."
    )
