"""
Violation Dataset Analyzer
============================
Validates dataset integrity, class distribution, split ratios,
and data quality for the traffic violation detection model.

Can be run standalone:
    python -m analysis.analyze_violation_dataset <dataset_path>
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


def analyze_violation_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of a YOLO-format violation detection dataset.

    Args:
        dataset_path: Path to the dataset root (containing data.yaml).

    Returns:
        Dict with analysis results matching DatasetAnalysisResponse schema.
    """
    root = Path(dataset_path)
    issues = []
    result = {
        "dataset_path": str(root),
        "dataset_type": "violation",
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
        issues.append("[ERROR] data.yaml not found in dataset root")
        result["issues"] = issues
        result["is_valid"] = False
        return result

    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    nc = data_config.get("nc", 0)
    names = data_config.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]

    if nc == 0:
        issues.append("[WARN] 'nc' (number of classes) is 0 or missing in data.yaml")
    if not names:
        issues.append("[WARN] 'names' is empty in data.yaml")
    if nc != len(names):
        issues.append(f"[WARN] nc={nc} doesn't match len(names)={len(names)}")

    # ── Analyze each split ──
    class_counter = Counter()
    total_images = 0
    total_labels = 0

    for split in ["train", "valid", "val", "test"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists():
            continue

        images = set()
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            images.update(img_dir.glob(ext))

        labels = set(lbl_dir.glob("*.txt")) if lbl_dir.exists() else set()

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        # Missing labels
        missing_labels = img_stems - lbl_stems
        if missing_labels:
            issues.append(
                f"[WARN] {split}: {len(missing_labels)} images without labels "
                f"(e.g., {list(missing_labels)[:3]})"
            )

        # Orphan labels
        orphan_labels = lbl_stems - img_stems
        if orphan_labels:
            issues.append(
                f"[WARN] {split}: {len(orphan_labels)} labels without images"
            )

        # Count classes from labels
        for lbl_file in labels:
            try:
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            cls_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
                            class_counter[cls_name] += 1

                            if cls_id >= nc and nc > 0:
                                issues.append(
                                    f"[WARN] {split}/{lbl_file.name}: class_id {cls_id} >= nc ({nc})"
                                )
            except Exception as e:
                issues.append(f"[ERROR] {split}/{lbl_file.name}: Parse error - {e}")

        # Empty labels
        empty_labels = 0
        for lbl_file in labels:
            if lbl_file.stat().st_size == 0:
                empty_labels += 1
        if empty_labels > 0:
            issues.append(f"[INFO] {split}: {empty_labels} empty label files (background images)")

        # Verify a few images are readable
        if Image:
            corrupt_count = 0
            for img_path in list(images)[:50]:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception:
                    corrupt_count += 1
            if corrupt_count > 0:
                issues.append(f"[ERROR] {split}: {corrupt_count} corrupt images (checked first 50)")

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
    result["issues"] = issues

    # Overall validity
    if any("[ERROR]" in issue for issue in issues):
        result["is_valid"] = False
    if total_images == 0:
        result["is_valid"] = False
        issues.append("[ERROR] No images found in dataset")

    # Print summary
    print("\n" + "=" * 60)
    print("VIOLATION DATASET ANALYSIS")
    print("=" * 60)
    print(f"Path: {root}")
    print(f"Classes: {nc}")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")

    print(f"\n{'Split Distribution':^40}")
    print("-" * 40)
    for split, counts in result["splits"].items():
        print(f"  {split:10s}: {counts['images']:6d} images, {counts['labels']:6d} labels")

    print(f"\n{'Class Distribution':^40}")
    print("-" * 40)
    max_count = max(class_counter.values()) if class_counter else 1
    for cls_name, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        map_val = count / max_count
        bar_fill = "#" * int(map_val * 30)
        bar_empty = "-" * (30 - int(map_val * 30))
        print(f"  {cls_name:25s} {bar_fill}{bar_empty} {count}")

    if issues:
        issue_count = len(issues)
        print(f"\n{'Issues Found (' + str(issue_count) + ')':^40}")
        print("-" * 40)
        for issue in issues:
            print(f"  {issue}")

    print(f"\n{'[OK] Dataset is VALID' if result['is_valid'] else '[ERROR] Dataset has ISSUES'}")
    print("=" * 60)

    return result


# CLI entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.analyze_violation_dataset <dataset_path>")
        print("Example: python -m analysis.analyze_violation_dataset ../datasets/combined-full")
        sys.exit(1)

    analyze_violation_dataset(sys.argv[1])
