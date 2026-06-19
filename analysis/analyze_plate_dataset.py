"""
Plate Dataset Analyzer
========================
Validates dataset integrity for the plate detection model.

Can be run standalone:
    python -m analysis.analyze_plate_dataset <dataset_path>
"""

import os
import sys
import yaml
from pathlib import Path
from collections import Counter
from typing import Dict, Any

try:
    from PIL import Image
except ImportError:
    Image = None


def analyze_plate_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Analyze a YOLO-format plate detection dataset.

    Returns:
        Dict with analysis results.
    """
    root = Path(dataset_path)
    issues = []
    result = {
        "dataset_path": str(root),
        "dataset_type": "plate",
        "total_images": 0,
        "total_labels": 0,
        "class_distribution": {},
        "issues": [],
        "splits": {},
        "is_valid": True,
    }

    # ── Check data.yaml ──
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        issues.append("❌ data.yaml not found")
        result["issues"] = issues
        result["is_valid"] = False
        return result

    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    nc = data_config.get("nc", 0)
    names = data_config.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    # ── Analyze splits ──
    class_counter = Counter()
    total_images = 0
    total_labels = 0
    bbox_widths = []
    bbox_heights = []

    for split in ["train", "valid", "val", "test"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists():
            continue

        images = set()
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.update(img_dir.glob(ext))

        labels = set(lbl_dir.glob("*.txt")) if lbl_dir.exists() else set()

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing = img_stems - lbl_stems
        if missing:
            issues.append(f"⚠️  {split}: {len(missing)} images without labels")

        # Analyze bounding boxes
        for lbl_file in labels:
            try:
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            w = float(parts[3])
                            h = float(parts[4])
                            cls_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
                            class_counter[cls_name] += 1
                            bbox_widths.append(w)
                            bbox_heights.append(h)

                            # Check for very small plates
                            if w < 0.01 or h < 0.005:
                                issues.append(
                                    f"⚠️  {split}/{lbl_file.name}: Very small bbox ({w:.4f}x{h:.4f})"
                                )
            except Exception as e:
                issues.append(f"❌ {split}/{lbl_file.name}: {e}")

        split_name = split if split != "val" else "valid"
        result["splits"][split_name] = {
            "images": len(images),
            "labels": len(labels),
        }
        total_images += len(images)
        total_labels += len(labels)

    result["total_images"] = total_images
    result["total_labels"] = total_labels
    result["class_distribution"] = dict(class_counter)

    # Bbox size analysis
    if bbox_widths:
        import statistics

        avg_w = statistics.mean(bbox_widths)
        avg_h = statistics.mean(bbox_heights)
        if avg_w < 0.05:
            issues.append(f"ℹ️  Average plate width is very small ({avg_w:.4f}). Consider higher resolution images.")
        if avg_h < 0.02:
            issues.append(f"ℹ️  Average plate height is very small ({avg_h:.4f}).")

    if total_images == 0:
        issues.append("❌ No images found")
        result["is_valid"] = False

    # Deduplicate issues (small bbox warnings can be noisy)
    seen = set()
    unique_issues = []
    for issue in issues:
        key = issue[:60]  # Group similar issues
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)
    result["issues"] = unique_issues[:50]  # Cap at 50

    if any("❌" in i for i in unique_issues):
        result["is_valid"] = False

    # Print summary
    print("\n" + "=" * 60)
    print("PLATE DATASET ANALYSIS")
    print("=" * 60)
    print(f"Path: {root}")
    print(f"Classes: {nc}")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")

    for split, counts in result["splits"].items():
        print(f"  {split:10s}: {counts['images']:6d} images, {counts['labels']:6d} labels")

    for cls_name, count in class_counter.items():
        print(f"  {cls_name}: {count}")

    if unique_issues:
        print(f"\nIssues ({len(unique_issues)}):")
        for issue in unique_issues[:20]:
            print(f"  {issue}")

    print(f"\n{'✅ VALID' if result['is_valid'] else '❌ HAS ISSUES'}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.analyze_plate_dataset <dataset_path>")
        sys.exit(1)

    analyze_plate_dataset(sys.argv[1])
