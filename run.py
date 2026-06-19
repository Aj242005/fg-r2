"""
Traffic Violation Detection — CLI Entry Point
================================================
Run detection on images, videos, webcam, or start the API server.

Usage:
    # Analyze a single image
    python run.py --source image.jpg

    # Analyze a video
    python run.py --source video.mp4

    # Analyze webcam
    python run.py --source 0

    # Start the FastAPI server
    python run.py --server

    # Analyze a dataset
    python run.py --analyze-dataset ../datasets/combined-full --type violation

    # Custom model paths
    python run.py --source image.jpg --violation-model models/violation_model.pt --plate-model models/plate_model.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_detection(args):
    """Run detection on image/video/webcam."""
    import cv2
    from pipeline import ViolationPipeline
    from visualizer import annotate_frame
    import config

    print("=" * 60)
    print("🚦 TRAFFIC VIOLATION DETECTION SYSTEM")
    print("=" * 60)

    # Initialize pipeline
    pipeline = ViolationPipeline(
        violation_model_path=args.violation_model,
        plate_model_path=args.plate_model,
        enable_plate_reader=not args.no_plate,
    )

    source = args.source

    # ── IMAGE MODE ──
    if source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        print(f"\n📸 Analyzing image: {source}")
        report = pipeline.process_image(source)
        report_dict = report.to_dict()

        # Print report
        print(f"\n{'VIOLATION REPORT':^60}")
        print("=" * 60)
        print(json.dumps(report_dict, indent=2))

        # Save annotated image
        frame = cv2.imread(source)
        annotated = annotate_frame(frame, report_dict)
        out_path = str(config.OUTPUT_DIR / f"annotated_{Path(source).name}")
        cv2.imwrite(out_path, annotated)
        print(f"\n✅ Annotated image saved: {out_path}")

        # Save JSON report
        json_path = str(config.OUTPUT_DIR / f"report_{Path(source).stem}.json")
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"✅ JSON report saved: {json_path}")

    # ── VIDEO MODE ──
    elif source.lower().endswith((".mp4", ".avi", ".mkv", ".mov")) or source.isdigit():
        if source.isdigit():
            source = int(source)
            print(f"\n📹 Analyzing webcam: device {source}")
        else:
            print(f"\n📹 Analyzing video: {source}")

        reports = pipeline.process_video(
            str(source),
            stride=args.stride,
            max_frames=args.max_frames,
        )

        # Summary
        report_dicts = [r.to_dict() for r in reports]
        violation_counts = {}
        for r in report_dicts:
            for v in r.get("violations", []):
                vt = v["type"]
                violation_counts[vt] = violation_counts.get(vt, 0) + 1

        print(f"\n{'VIDEO ANALYSIS SUMMARY':^60}")
        print("=" * 60)
        print(f"Frames with violations: {len(reports)}")
        print(f"Total violations: {sum(violation_counts.values())}")
        for vt, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
            print(f"  {vt}: {count}")

        # Save reports
        json_path = str(config.OUTPUT_DIR / "video_report.json")
        with open(json_path, "w") as f:
            json.dump(report_dicts, f, indent=2)
        print(f"\n✅ Report saved: {json_path}")

    # ── DIRECTORY MODE ──
    elif Path(source).is_dir():
        print(f"\n📁 Analyzing directory: {source}")
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [
            f for f in Path(source).iterdir()
            if f.suffix.lower() in image_exts
        ]
        print(f"Found {len(images)} images")

        all_reports = []
        for img_path in images:
            report = pipeline.process_image(str(img_path))
            if report.has_violations:
                all_reports.append(report.to_dict())
                print(f"  🚨 {img_path.name}: {report.to_dict()['violation_count']} violations")
            else:
                print(f"  ✅ {img_path.name}: clean")

        json_path = str(config.OUTPUT_DIR / "batch_report.json")
        with open(json_path, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\n✅ Batch report saved: {json_path}")

    else:
        print(f"❌ Unsupported source: {source}")
        sys.exit(1)

    pipeline.shutdown()


def run_server(args):
    """Start the FastAPI server."""
    import uvicorn
    import config

    print("🚀 Starting Traffic Violation Detection Server...")
    uvicorn.run(
        "server.app:app",
        host=args.host or config.SERVER_HOST,
        port=args.port or config.SERVER_PORT,
        reload=args.reload,
        log_level="info",
    )


def run_dataset_analysis(args):
    """Run dataset analysis."""
    from analysis.analyze_violation_dataset import analyze_violation_dataset
    from analysis.analyze_plate_dataset import analyze_plate_dataset

    if args.type == "violation":
        result = analyze_violation_dataset(args.analyze_dataset)
    else:
        result = analyze_plate_dataset(args.analyze_dataset)

    # Save result
    import config
    json_path = str(config.OUTPUT_DIR / f"dataset_analysis_{args.type}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Analysis saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --source image.jpg              # Analyze image
  python run.py --source video.mp4 --stride 3   # Analyze video (every 3rd frame)
  python run.py --source 0                      # Webcam
  python run.py --server                        # Start API server
  python run.py --server --port 5000            # Server on custom port
  python run.py --analyze-dataset ../datasets/combined-full --type violation
        """,
    )

    # Mode selection
    parser.add_argument("--source", type=str, help="Image/video/webcam/directory path")
    parser.add_argument("--server", action="store_true", help="Start FastAPI server")
    parser.add_argument("--analyze-dataset", type=str, help="Path to dataset to analyze")

    # Model paths
    parser.add_argument("--violation-model", type=str, default=None, help="Violation model .pt path")
    parser.add_argument("--plate-model", type=str, default=None, help="Plate model .pt path")
    parser.add_argument("--no-plate", action="store_true", help="Disable plate reading")

    # Video options
    parser.add_argument("--stride", type=int, default=1, help="Video frame stride")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")

    # Server options
    parser.add_argument("--host", type=str, default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")

    # Dataset analysis
    parser.add_argument("--type", type=str, default="violation", choices=["violation", "plate"])

    args = parser.parse_args()

    if args.server:
        run_server(args)
    elif args.analyze_dataset:
        run_dataset_analysis(args)
    elif args.source:
        run_detection(args)
    else:
        parser.print_help()
        print("\n❌ Please specify --source, --server, or --analyze-dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()
