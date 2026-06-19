"""
================================================================
KAGGLE TRAINING SCRIPT — MODEL 1: TRAFFIC VIOLATION DETECTION
================================================================
Train a YOLOv8 model for comprehensive traffic violation detection.
Covers 21 classes: vehicles, helmets, traffic lights, stop lines.

USAGE ON KAGGLE:
    1. Create a new Kaggle notebook with GPU enabled (P100 or T4)
    2. Add your dataset (uploaded via Kaggle Datasets)
    3. Copy-paste this entire script into a cell and run
    4. Download the resulting best.pt
    5. Rename to violation_model.pt and place in code/models/

DATASET STRUCTURE EXPECTED:
    your-dataset/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── valid/
    │   └── test/
    └── labels/
        ├── train/
        ├── valid/
        └── test/
================================================================
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
import glob

# ════════════════════════════════════════════════════════════════
# CONFIGURATION — Modify these settings as needed
# ════════════════════════════════════════════════════════════════

class TrainingConfig:
    """Training configuration for the violation detection model."""

    # ── Model Architecture ──
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium),
    #          yolov8l.pt (large), yolov8x.pt (extra-large)
    # Recommendation: yolov8m.pt for balanced speed/accuracy
    MODEL_ARCH = "yolov8m.pt"

    # ── Training Hyperparameters ──
    # EPOCHS: Reduced from 150 → 100 to fit within Kaggle's 12h free limit
    # (10h target = 12h limit − 2h margin). At ~5 min/epoch on a T4 with
    # ~12k images, 100 epochs ≈ 8.3h training + ~40 min overhead ≈ 9h total.
    # Early stopping (PATIENCE=30) will halt sooner if the model converges.
    EPOCHS = 100
    BATCH_SIZE = 16          # Reduce to 8 if OOM on T4
    IMG_SIZE = 640           # Use 1280 for higher accuracy (slower)
    OPTIMIZER = "AdamW"
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005
    PATIENCE = 30            # Early stopping patience (reduced from 50 to match fewer epochs)

    # ── Data Augmentation ──
    MOSAIC = 1.0
    MIXUP = 0.15
    COPY_PASTE = 0.1
    DEGREES = 10.0
    TRANSLATE = 0.2
    SCALE = 0.9
    SHEAR = 2.0
    PERSPECTIVE = 0.0001
    FLIPUD = 0.0
    FLIPLR = 0.5
    HSV_H = 0.015
    HSV_S = 0.7
    HSV_V = 0.4
    ERASING = 0.3

    # ── Advanced ──
    DEVICE = 0               # GPU 0
    WORKERS = 4
    CACHE = True             # Cache images in RAM for speed
    AMP = True               # Automatic mixed precision
    COS_LR = True            # Cosine annealing LR scheduler
    CLOSE_MOSAIC = 10        # Disable mosaic last N epochs (reduced with epoch count)
    SAVE_PERIOD = 5          # Save checkpoint every N epochs (more frequent in fewer epochs)
    LABEL_SMOOTHING = 0.01

    # ── Project ──
    PROJECT = "traffic_violation_runs"
    NAME = "violation_model_v1"

config = TrainingConfig()

# ════════════════════════════════════════════════════════════════
# STEP 1: DETECT ENVIRONMENT & FIND DATASET
# ════════════════════════════════════════════════════════════════

print("=" * 70)
print("TRAFFIC VIOLATION DETECTION — MODEL TRAINING")
print("=" * 70)

IS_KAGGLE = os.path.exists("/kaggle/working")
IS_COLAB = os.path.exists("/content")

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'}")

# Install dependencies
if IS_KAGGLE or IS_COLAB:
    os.system("pip install -q ultralytics opencv-python-headless pyyaml matplotlib seaborn")

from ultralytics import YOLO

# Auto-detect dataset path
def find_dataset():
    """Search common locations for the dataset."""
    search_paths = []

    if IS_KAGGLE:
        # Search Kaggle input directories
        input_dir = Path("/kaggle/input")
        if input_dir.exists():
            for d in input_dir.iterdir():
                if d.is_dir():
                    search_paths.append(d)
                    # Check subdirectories too
                    for sd in d.iterdir():
                        if sd.is_dir():
                            search_paths.append(sd)

    # Common paths
    search_paths.extend([
        Path("combined-full"),
        Path("combined"),
        Path("dataset"),
        Path("."),
    ])

    for path in search_paths:
        yaml_file = path / "data.yaml"
        if yaml_file.exists():
            print(f"✅ Found dataset at: {path}")
            return path

    raise FileNotFoundError(
        "Dataset not found! Please ensure data.yaml is in one of the input directories."
    )

dataset_path = find_dataset()

# ════════════════════════════════════════════════════════════════
# STEP 2: PREPARE data.yaml WITH CORRECT PATHS
# ════════════════════════════════════════════════════════════════

original_yaml = dataset_path / "data.yaml"
with open(original_yaml, "r") as f:
    data_config = yaml.safe_load(f)

# Fix paths to absolute
abs_path = dataset_path.absolute()
data_config["path"] = str(abs_path)
data_config["train"] = "images/train"
data_config["val"] = "images/valid"
data_config["test"] = "images/test"

# Save corrected yaml
corrected_yaml = Path("data_corrected.yaml")
with open(corrected_yaml, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"\n📋 Dataset Configuration:")
print(f"   Path: {abs_path}")
print(f"   Classes: {data_config['nc']}")
print(f"   Names: {data_config['names']}")

# Verify splits exist
for split in ["train", "valid", "test"]:
    img_dir = abs_path / "images" / split
    if img_dir.exists():
        count = len(list(img_dir.glob("*")))
        print(f"   {split}: {count} images")
    else:
        # Try 'val' instead of 'valid'
        alt_dir = abs_path / "images" / ("val" if split == "valid" else split)
        if alt_dir.exists():
            count = len(list(alt_dir.glob("*")))
            print(f"   {split} (as 'val'): {count} images")
            data_config["val"] = "images/val"
            with open(corrected_yaml, "w") as f:
                yaml.dump(data_config, f, default_flow_style=False)
        else:
            print(f"   ⚠️  {split}: NOT FOUND")

# ════════════════════════════════════════════════════════════════
# STEP 3: TRAIN THE MODEL
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

    # Optimizer
    optimizer=config.OPTIMIZER,
    lr0=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    patience=config.PATIENCE,
    cos_lr=config.COS_LR,
    label_smoothing=config.LABEL_SMOOTHING,

    # Augmentation
    augment=True,
    mosaic=config.MOSAIC,
    mixup=config.MIXUP,
    copy_paste=config.COPY_PASTE,
    degrees=config.DEGREES,
    translate=config.TRANSLATE,
    scale=config.SCALE,
    shear=config.SHEAR,
    perspective=config.PERSPECTIVE,
    flipud=config.FLIPUD,
    fliplr=config.FLIPLR,
    hsv_h=config.HSV_H,
    hsv_s=config.HSV_S,
    hsv_v=config.HSV_V,
    erasing=config.ERASING,
    close_mosaic=config.CLOSE_MOSAIC,

    # Performance
    cache=config.CACHE,
    amp=config.AMP,

    # Saving
    save=True,
    save_period=config.SAVE_PERIOD,
    val=True,
    plots=True,
    save_json=True,
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

# Per-class metrics
if hasattr(metrics.box, "maps"):
    print(f"\n{'Per-Class mAP50':^50}")
    print("-" * 50)
    names = data_config.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    for i, map_val in enumerate(metrics.box.maps):
        name = names[i] if i < len(names) else f"class_{i}"
        bar = "█" * int(map_val * 30) + "░" * (30 - int(map_val * 30))
        print(f"   {name:25s} {bar} {map_val:.4f}")

# ════════════════════════════════════════════════════════════════
# STEP 5: EXPORT & PACKAGE
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPORTING MODEL")
print("=" * 70)

results_dir = Path(config.PROJECT) / config.NAME
weights_dir = results_dir / "weights"
best_pt = weights_dir / "best.pt"

if best_pt.exists():
    # Copy to easy download location
    download_dir = Path("model_download")
    download_dir.mkdir(exist_ok=True)

    shutil.copy2(best_pt, download_dir / "violation_model.pt")
    shutil.copy2(str(corrected_yaml), download_dir / "data.yaml")

    # Export to ONNX
    try:
        model.export(format="onnx")
        onnx_path = best_pt.with_suffix(".onnx")
        if onnx_path.exists():
            shutil.copy2(onnx_path, download_dir / "violation_model.onnx")
    except Exception as e:
        print(f"   ⚠️  ONNX export failed: {e}")

    print(f"\n✅ Model saved to: {download_dir.absolute()}")
    print(f"   📁 violation_model.pt  — Drop this into code/models/")
    print(f"   📁 data.yaml           — Dataset configuration")

    if IS_KAGGLE:
        shutil.copytree(str(download_dir), "/kaggle/working/model_download", dirs_exist_ok=True)
        print("\n📦 Files copied to /kaggle/working/model_download")
        print("   Download from the Output tab →")
else:
    print("❌ best.pt not found! Check training logs for errors.")

# ════════════════════════════════════════════════════════════════
# STEP 6: FINAL SUMMARY
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("🎯 TRAINING COMPLETE — SUMMARY")
print("=" * 70)
print(f"   Model:      {config.MODEL_ARCH}")
print(f"   Epochs:     {config.EPOCHS}")
print(f"   mAP50:      {metrics.box.map50:.4f}")
print(f"   mAP50-95:   {metrics.box.map:.4f}")
print(f"   Precision:  {metrics.box.mp:.4f}")
print(f"   Recall:     {metrics.box.mr:.4f}")
print()
print("📋 NEXT STEPS:")
print("   1. Download 'violation_model.pt' from the Output tab")
print("   2. Place it in: code/models/violation_model.pt")
print("   3. Start the server: python -m server.app")
print("=" * 70)
