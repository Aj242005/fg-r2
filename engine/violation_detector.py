"""
Violation Detector — Model 1 Inference Wrapper
================================================
Loads the YOLOv8 violation detection model and returns structured detections.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class Detection:
    """Single detection result from the violation model."""

    def __init__(
        self,
        bbox: List[float],
        class_id: int,
        class_name: str,
        confidence: float,
    ):
        self.bbox = bbox          # [x1, y1, x2, y2]
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(v, 2) for v in self.bbox],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
        }

    def __repr__(self):
        return (
            f"Detection({self.class_name}, conf={self.confidence:.2f}, "
            f"bbox=[{self.bbox[0]:.0f},{self.bbox[1]:.0f},{self.bbox[2]:.0f},{self.bbox[3]:.0f}])"
        )


class ViolationDetector:
    """
    Wrapper around the YOLOv8 violation detection model.

    Usage:
        detector = ViolationDetector()
        detections = detector.detect(frame)
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

        # Resolve model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = config.get_violation_model_path()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Violation model not found at {self.model_path}")

        # Load model
        self.model = YOLO(str(self.model_path))
        self.device = device if device != "auto" else config.DEVICE
        self.class_names = config.VIOLATION_CLASSES
        self._loaded = True

        print(f"[ViolationDetector] Loaded model from {self.model_path}")
        print(f"[ViolationDetector] Classes: {len(self.class_names)}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(
        self,
        frame: np.ndarray,
        conf: float = None,
        iou: float = None,
        img_size: int = None,
    ) -> List[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: BGR numpy array (from cv2).
            conf: Confidence threshold override.
            iou: NMS IoU threshold override.
            img_size: Inference image size override.

        Returns:
            List of Detection objects.
        """
        conf = conf or config.VIOLATION_CONF_THRESHOLD
        iou = iou or config.NMS_IOU_THRESHOLD
        img_size = img_size or config.INFERENCE_IMG_SIZE

        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            max_det=config.MAX_DETECTIONS,
            verbose=False,
            device=self.device,
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()

                    class_name = self.class_names.get(class_id, f"class_{class_id}")

                    detections.append(
                        Detection(
                            bbox=bbox,
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                        )
                    )

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        conf: float = None,
    ) -> List[List[Detection]]:
        """Run inference on a batch of frames."""
        return [self.detect(frame, conf=conf) for frame in frames]

    # ----- Convenience filters -----

    @staticmethod
    def filter_vehicles(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id in config.VEHICLE_CLASSES]

    @staticmethod
    def filter_two_wheelers(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id in config.TWO_WHEELER_CLASSES]

    @staticmethod
    def filter_three_wheelers(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id in config.THREE_WHEELER_CLASSES]

    @staticmethod
    def filter_four_wheelers(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id in config.FOUR_WHEELER_CLASSES]

    @staticmethod
    def filter_riders(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id in config.RIDER_CLASSES]

    @staticmethod
    def filter_helmets_absent(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id in config.HELMET_ABSENT_CLASSES]

    @staticmethod
    def filter_helmets_present(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id in config.HELMET_PRESENT_CLASSES]

    @staticmethod
    def filter_traffic_lights(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id in config.TRAFFIC_LIGHT_CLASSES]

    @staticmethod
    def filter_stop_lines(detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_id == config.STOP_LINE_CLASS]
