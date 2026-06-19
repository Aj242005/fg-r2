"""
Plate Reader — Model 2 Inference + OCR
========================================
Loads the YOLOv8 plate detection model, crops detected plates,
and runs EasyOCR to extract plate text.
Runs in parallel with the violation detector.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import re
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import easyocr
except ImportError:
    easyocr = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class PlateResult:
    """Single plate detection + OCR result."""

    def __init__(
        self,
        bbox: List[float],
        plate_text: str,
        detection_confidence: float,
        ocr_confidence: float,
    ):
        self.bbox = bbox
        self.plate_text = plate_text
        self.detection_confidence = detection_confidence
        self.ocr_confidence = ocr_confidence
        self.confidence = (detection_confidence + ocr_confidence) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(v, 2) for v in self.bbox],
            "plate_text": self.plate_text,
            "detection_confidence": round(self.detection_confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 4),
            "combined_confidence": round(self.confidence, 4),
        }

    def __repr__(self):
        return f"PlateResult('{self.plate_text}', conf={self.confidence:.2f})"


class PlateReader:
    """
    Number plate detector + OCR reader.

    Pipeline:
        frame → YOLOv8 (detect plate bbox) → crop → preprocess → EasyOCR → text

    Usage:
        reader = PlateReader()
        plates = reader.read_plates(frame)
    """

    # Indian license plate pattern: XX 00 XX 0000
    INDIAN_PLATE_PATTERN = re.compile(
        r"[A-Z]{2}\s*\d{1,2}\s*[A-Z]{0,3}\s*\d{1,4}", re.IGNORECASE
    )

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

        # Resolve model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = config.get_plate_model_path()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Plate model not found at {self.model_path}")

        # Load YOLO model for plate detection
        self.model = YOLO(str(self.model_path))
        self.device = device if device != "auto" else config.DEVICE

        # Initialize EasyOCR reader
        if easyocr is None:
            print("[PlateReader] WARNING: easyocr not installed. OCR disabled.")
            self.ocr_reader = None
        else:
            print("[PlateReader] Initializing EasyOCR (first load may download models)...")
            self.ocr_reader = easyocr.Reader(["en"], gpu=self.device != "cpu")

        self._loaded = True
        print(f"[PlateReader] Loaded plate model from {self.model_path}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _preprocess_plate_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess a plate crop for better OCR accuracy.
        - Resize to standard height
        - Convert to grayscale
        - Apply adaptive thresholding
        - Denoise
        """
        # Resize to fixed height while maintaining aspect ratio
        target_h = 100
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive thresholding for clear text
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def _clean_plate_text(self, raw_text: str) -> str:
        """
        Clean and normalize OCR output for Indian license plates.
        Applies character corrections common in plate OCR.
        """
        text = raw_text.upper().strip()

        # Common OCR misreads
        replacements = {
            "O": "0", "I": "1", "S": "5", "Z": "2",
            "B": "8", "G": "6", "T": "7",
        }

        # Only apply number corrections in positions that should be digits
        # For Indian plates: XX 00 XX(X) 0000
        # Keep letters as-is in letter positions
        cleaned = re.sub(r"[^A-Z0-9]", "", text)

        return cleaned

    def _extract_text(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on a plate crop and return (text, confidence).
        """
        if self.ocr_reader is None:
            return ("OCR_UNAVAILABLE", 0.0)

        try:
            preprocessed = self._preprocess_plate_crop(crop)
            results = self.ocr_reader.readtext(preprocessed, detail=1)

            if not results:
                # Try on original crop without preprocessing
                results = self.ocr_reader.readtext(crop, detail=1)

            if not results:
                return ("UNREADABLE", 0.0)

            # Combine all text segments
            full_text = " ".join([r[1] for r in results])
            avg_conf = np.mean([r[2] for r in results])

            cleaned = self._clean_plate_text(full_text)
            return (cleaned, float(avg_conf))

        except Exception as e:
            print(f"[PlateReader] OCR error: {e}")
            return ("ERROR", 0.0)

    def read_plates(
        self,
        frame: np.ndarray,
        conf: float = None,
    ) -> List[PlateResult]:
        """
        Detect plates in frame and read text via OCR.

        Args:
            frame: BGR numpy array.
            conf: Detection confidence threshold.

        Returns:
            List of PlateResult with bbox, plate text, and confidences.
        """
        conf = conf or config.PLATE_CONF_THRESHOLD

        # Step 1: Detect plate bounding boxes
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=config.NMS_IOU_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            max_det=20,
            verbose=False,
            device=self.device,
        )

        plate_results = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                h, w = frame.shape[:2]

                for box in boxes:
                    det_conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()

                    # Clamp bbox to image boundaries
                    x1 = max(0, int(bbox[0]))
                    y1 = max(0, int(bbox[1]))
                    x2 = min(w, int(bbox[2]))
                    y2 = min(h, int(bbox[3]))

                    if x2 - x1 < 10 or y2 - y1 < 5:
                        continue  # too small

                    # Step 2: Crop plate region
                    plate_crop = frame[y1:y2, x1:x2]

                    # Step 3: OCR
                    plate_text, ocr_conf = self._extract_text(plate_crop)

                    plate_results.append(
                        PlateResult(
                            bbox=bbox,
                            plate_text=plate_text,
                            detection_confidence=det_conf,
                            ocr_confidence=ocr_conf,
                        )
                    )

        return plate_results
