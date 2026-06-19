"""
================================================================
KAGGLE TRAINING SCRIPT — MODEL 2: NUMBER PLATE DETECTION + OCR
================================================================
Train a YOLOv8 model for license plate detection.
After detection, EasyOCR is used for text extraction (at inference time).

RECOMMENDED KAGGLE DATASETS FOR PLATE DETECTION:
    1. "License Plate Recognition" — search Kaggle for:
       - "license-plate-detection" (YOLO format)
       - "car-plate-detection" (YOLO format)
       - "automatic-number-plate-recognition"
    2. Or use Roboflow:
       - Search "license plate" on universe.roboflow.com
       - Download in YOLOv8 format

USAGE ON KAGGLE:
    1. Create a new Kaggle notebook with GPU enabled
    2. Add a license plate dataset (YOLO format)
    3. Copy-paste this entire script into a cell and run
    4. Download the resulting best.pt
    5. Rename to plate_model.pt and place in code/models/

DATASET STRUCTURE EXPECTED:
    your-plate-dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── valid/ (or val/)
    │   └── test/  (optional)
    └── labels/
        ├── train/
        ├── valid/ (or val/)
        └── test/  (optional)
================================================================
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════

class PlateTrainingConfig:
    """Training configuration for the plate detection model."""

    # ── Model Architecture ──
    # Plates are small objects → yolov8s or yolov8m works well
    MODEL_ARCH = "yolov8s.pt"

    # ── Training Hyperparameters ──
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 640
    OPTIMIZER = "AdamW"
    LEARNING_RATE = 0.001
    PATIENCE = 40

    # ── Data Augmentation ──
    # Plates need specific augmentation (don't flip vertically, moderate rotation)
    MOSAIC = 1.0
    MIXUP = 0.1
    DEGREES = 5.0           # Less rotation (plates need to be readable)
    TRANSLATE = 0.15
    SCALE = 0.5
    SHEAR = 1.0
    FLIPLR = 0.5
    FLIPUD = 0.0            # Never flip vertically
    HSV_H = 0.015
    HSV_S = 0.5
    HSV_V = 0.4
    ERASING = 0.2

    # ── Advanced ──
    DEVICE = 0
    WORKERS = 4
    CACHE = True
    AMP = True
    COS_LR = True
    CLOSE_MOSAIC = 10
    SAVE_PERIOD = 10

    # ── Project ──
    PROJECT = "plate_detection_runs"
    NAME = "plate_model_v1"

config = PlateTrainingConfig()

# ════════════════════════════════════════════════════════════════
# STEP 1: SETUP
# ════════════════════════════════════════════════════════════════

print("=" * 70)
print("NUMBER PLATE DETECTION — MODEL TRAINING")
print("=" * 70)

IS_KAGGLE = os.path.exists("/kaggle/working")
IS_COLAB = os.path.exists("/content")

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'}")

# Install dependencies
if IS_KAGGLE or IS_COLAB:
    os.system("pip install -q ultralytics opencv-python-headless pyyaml easyocr")

from ultralytics import YOLO

# Auto-detect dataset
def find_plate_dataset():
    """Search for plate dataset."""
    search_paths = []

    if IS_KAGGLE:
        input_dir = Path("/kaggle/input")
        if input_dir.exists():
            for d in input_dir.iterdir():
                if d.is_dir():
                    search_paths.append(d)
                    for sd in d.iterdir():
                        if sd.is_dir():
                            search_paths.append(sd)

    search_paths.extend([
        Path("plate_dataset"),
        Path("license-plate"),
        Path("dataset"),
        Path("."),
    ])

    for path in search_paths:
        yaml_file = path / "data.yaml"
        if yaml_file.exists():
            print(f"✅ Found dataset at: {path}")
            return path

    raise FileNotFoundError(
        "Plate dataset not found! Please add a license plate dataset in YOLO format.\n"
        "Recommended: Search Kaggle for 'license-plate-detection' or\n"
        "             Download from Roboflow in YOLOv8 format."
    )

dataset_path = find_plate_dataset()

# ════════════════════════════════════════════════════════════════
# STEP 2: PREPARE data.yaml
# ════════════════════════════════════════════════════════════════

original_yaml = dataset_path / "data.yaml"
with open(original_yaml, "r") as f:
    data_config = yaml.safe_load(f)

abs_path = dataset_path.absolute()
data_config["path"] = str(abs_path)

# Normalize split paths
for key, options in [("train", ["images/train"]), ("val", ["images/valid", "images/val"])]:
    for opt in options:
        if (abs_path / opt).exists():
            data_config[key] = opt
            break

if (abs_path / "images/test").exists():
    data_config["test"] = "images/test"

corrected_yaml = Path("plate_data_corrected.yaml")
with open(corrected_yaml, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"\n📋 Dataset Configuration:")
print(f"   Path: {abs_path}")
print(f"   Classes: {data_config.get('nc', '?')}")
print(f"   Names: {data_config.get('names', '?')}")

# Count images per split
for split_key, split_paths in [("train", ["images/train"]), ("val", ["images/valid", "images/val"]), ("test", ["images/test"])]:
    for sp in split_paths:
        d = abs_path / sp
        if d.exists():
            count = len(list(d.glob("*")))
            print(f"   {split_key}: {count} images")
            break

# ════════════════════════════════════════════════════════════════
# STEP 3: TRAIN
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

model = YOLO(config.MODEL_ARCH)

results = model.train(
    data=str(corrected_yaml),
    epochs=config.EPOCHS,
    batch=config.BATCH_SIZE,
    imgsz=config.IMG_SIZE,
    device=config.DEVICE,
    workers=config.WORKERS,
    project=config.PROJECT,
    name=config.NAME,
    exist_ok=True,
    pretrained=True,

    optimizer=config.OPTIMIZER,
    lr0=config.LEARNING_RATE,
    patience=config.PATIENCE,
    cos_lr=config.COS_LR,

    augment=True,
    mosaic=config.MOSAIC,
    mixup=config.MIXUP,
    degrees=config.DEGREES,
    translate=config.TRANSLATE,
    scale=config.SCALE,
    shear=config.SHEAR,
    fliplr=config.FLIPLR,
    flipud=config.FLIPUD,
    hsv_h=config.HSV_H,
    hsv_s=config.HSV_S,
    hsv_v=config.HSV_V,
    erasing=config.ERASING,
    close_mosaic=config.CLOSE_MOSAIC,

    cache=config.CACHE,
    amp=config.AMP,
    save=True,
    save_period=config.SAVE_PERIOD,
    val=True,
    plots=True,
)

print("\n✅ Training complete!")

# ════════════════════════════════════════════════════════════════
# STEP 4: EVALUATE
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

metrics = model.val()

print(f"\n📊 Validation Metrics:")
print(f"   mAP50:     {metrics.box.map50:.4f}")
print(f"   mAP50-95:  {metrics.box.map:.4f}")
print(f"   Precision:  {metrics.box.mp:.4f}")
print(f"   Recall:     {metrics.box.mr:.4f}")

# ════════════════════════════════════════════════════════════════
# STEP 5: TEST OCR PIPELINE
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TESTING OCR PIPELINE")
print("=" * 70)

try:
    import easyocr
    import cv2
    import numpy as np

    reader = easyocr.Reader(["en"], gpu=True)

    # Find test images
    test_dir = None
    for d in [abs_path / "images/test", abs_path / "images/valid", abs_path / "images/val"]:
        if d.exists():
            test_dir = d
            break

    if test_dir:
        test_images = list(test_dir.glob("*.jpg"))[:5]

        for img_path in test_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Detect plates
            det_results = model.predict(source=img, conf=0.3, verbose=False)
            if det_results and det_results[0].boxes is not None:
                for box in det_results[0].boxes:
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    if crop.size == 0:
                        continue

                    # Run OCR
                    ocr_results = reader.readtext(crop)
                    if ocr_results:
                        text = " ".join([r[1] for r in ocr_results])
                        conf = np.mean([r[2] for r in ocr_results])
                        print(f"   📸 {img_path.name}: Plate='{text}' (conf={conf:.2f})")
                    else:
                        print(f"   📸 {img_path.name}: Plate detected but OCR failed")
            else:
                print(f"   📸 {img_path.name}: No plates detected")
    else:
        print("   ⚠️  No test images found for OCR testing")

except ImportError:
    print("   ⚠️  easyocr not available for testing. It will be used at inference time.")

# ════════════════════════════════════════════════════════════════
# STEP 6: EXPORT & PACKAGE
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPORTING MODEL")
print("=" * 70)

results_dir = Path(config.PROJECT) / config.NAME
best_pt = results_dir / "weights" / "best.pt"

if best_pt.exists():
    download_dir = Path("plate_model_download")
    download_dir.mkdir(exist_ok=True)

    shutil.copy2(best_pt, download_dir / "plate_model.pt")
    shutil.copy2(str(corrected_yaml), download_dir / "data.yaml")

    try:
        model.export(format="onnx")
        onnx_path = best_pt.with_suffix(".onnx")
        if onnx_path.exists():
            shutil.copy2(onnx_path, download_dir / "plate_model.onnx")
    except Exception as e:
        print(f"   ⚠️  ONNX export failed: {e}")

    print(f"\n✅ Model saved to: {download_dir.absolute()}")
    print(f"   📁 plate_model.pt  — Drop this into code/models/")

    if IS_KAGGLE:
        shutil.copytree(str(download_dir), "/kaggle/working/plate_model_download", dirs_exist_ok=True)
        print("\n📦 Files copied to /kaggle/working/plate_model_download")
        print("   Download from the Output tab →")
else:
    print("❌ best.pt not found!")

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("🎯 PLATE MODEL TRAINING COMPLETE")
print("=" * 70)
print(f"   Model:      {config.MODEL_ARCH}")
print(f"   Epochs:     {config.EPOCHS}")
print(f"   mAP50:      {metrics.box.map50:.4f}")
print(f"   mAP50-95:   {metrics.box.map:.4f}")
print()
print("📋 NEXT STEPS:")
print("   1. Download 'plate_model.pt' from the Output tab")
print("   2. Place it in: code/models/plate_model.pt")
print("   3. The server will auto-load it on startup")
print("=" * 70)
