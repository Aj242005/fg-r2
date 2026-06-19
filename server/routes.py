"""
API Routes for Traffic Violation Detection Server
===================================================
Endpoints for image, video, stream analysis, health check, and dataset validation.
"""

import os
import time
import json
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline import ViolationPipeline
from visualizer import annotate_frame
from server.schemas import (
    ImageAnalysisResponse,
    VideoAnalysisResponse,
    HealthResponse,
    DatasetAnalysisResponse,
)


router = APIRouter()

# ── Global pipeline instance (initialized in app.py startup) ──
_pipeline: Optional[ViolationPipeline] = None
_start_time: float = time.time()
_executor = ThreadPoolExecutor(max_workers=2)


def get_pipeline() -> ViolationPipeline:
    """Get the global pipeline instance."""
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Ensure models are loaded.",
        )
    return _pipeline


def set_pipeline(pipeline: ViolationPipeline):
    """Set the global pipeline instance."""
    global _pipeline
    _pipeline = pipeline


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check server health and model status."""
    pipeline = None
    try:
        pipeline = get_pipeline()
    except HTTPException:
        pass

    return HealthResponse(
        status="ok" if pipeline else "no_models",
        violation_model_loaded=pipeline.violation_detector.is_loaded if pipeline else False,
        plate_model_loaded=(
            pipeline.plate_reader.is_loaded if pipeline and pipeline.plate_reader else False
        ),
        violation_classes=len(config.VIOLATION_CLASSES),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


# ============================================================================
# IMAGE ANALYSIS
# ============================================================================

@router.post("/analyze/image", response_model=ImageAnalysisResponse, tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    return_annotated: bool = Query(True, description="Return annotated image"),
):
    """
    Analyze a single image for traffic violations.

    Upload an image and receive a JSON report with all detected violations,
    number plates, and confidence scores.
    """
    pipeline = get_pipeline()

    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run analysis in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(_executor, pipeline.process_frame, frame)
    report_dict = report.to_dict()

    # Generate annotated image
    annotated_url = None
    if return_annotated:
        annotated = annotate_frame(frame, report_dict)
        out_filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
        out_path = config.OUTPUT_DIR / out_filename
        cv2.imwrite(str(out_path), annotated)
        annotated_url = f"/output/{out_filename}"

    return ImageAnalysisResponse(
        success=True,
        report=report_dict,
        annotated_image_url=annotated_url,
    )


# ============================================================================
# VIDEO ANALYSIS
# ============================================================================

@router.post("/analyze/video", response_model=VideoAnalysisResponse, tags=["Analysis"])
async def analyze_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI)"),
    stride: int = Query(3, ge=1, description="Process every Nth frame"),
    max_frames: int = Query(None, description="Max frames to process"),
    return_annotated: bool = Query(False, description="Return annotated video"),
):
    """
    Analyze a video for traffic violations.

    Upload a video and receive a JSON report with violations per frame,
    number plates, and confidence scores.
    """
    pipeline = get_pipeline()

    # Save uploaded video to temp file
    suffix = Path(file.filename).suffix or ".mp4"
    temp_path = config.OUTPUT_DIR / f"upload_{uuid.uuid4().hex[:8]}{suffix}"

    with open(temp_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    try:
        # Process video
        loop = asyncio.get_event_loop()
        reports = await loop.run_in_executor(
            _executor,
            lambda: pipeline.process_video(
                str(temp_path), stride=stride, max_frames=max_frames
            ),
        )

        report_dicts = [r.to_dict() for r in reports]

        # Compute summary
        violation_types = {}
        for r in report_dicts:
            for v in r.get("violations", []):
                vt = v["type"]
                violation_types[vt] = violation_types.get(vt, 0) + 1

        cap = cv2.VideoCapture(str(temp_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        annotated_url = None
        if return_annotated and report_dicts:
            from visualizer import annotate_video

            out_filename = f"annotated_{uuid.uuid4().hex[:8]}.mp4"
            out_path = config.OUTPUT_DIR / out_filename
            await loop.run_in_executor(
                _executor,
                lambda: annotate_video(str(temp_path), str(out_path), report_dicts),
            )
            annotated_url = f"/output/{out_filename}"

        return VideoAnalysisResponse(
            success=True,
            total_frames_processed=total_frames,
            frames_with_violations=len(report_dicts),
            reports=report_dicts,
            annotated_video_url=annotated_url,
            summary={
                "violation_counts": violation_types,
                "total_violations": sum(violation_types.values()),
            },
        )
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


# ============================================================================
# LIVE STREAM ANALYSIS (WebSocket)
# ============================================================================

@router.websocket("/analyze/stream")
async def analyze_stream(websocket: WebSocket):
    """
    WebSocket endpoint for live stream analysis.

    Send JSON: {"stream_url": "rtsp://...", "stride": 5}
    Receive JSON: ViolationReport per analyzed frame
    """
    await websocket.accept()
    pipeline = get_pipeline()

    try:
        # Receive stream configuration
        data = await websocket.receive_text()
        config_data = json.loads(data)
        stream_url = config_data.get("stream_url", "")
        stride = config_data.get("stride", 5)

        if not stream_url:
            await websocket.send_json({"error": "No stream_url provided"})
            await websocket.close()
            return

        await websocket.send_json({"status": "connecting", "url": stream_url})

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            await websocket.send_json({"error": f"Cannot connect to stream: {stream_url}"})
            await websocket.close()
            return

        await websocket.send_json({"status": "streaming"})

        frame_id = 0
        pipeline.tracker.reset()

        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_json({"status": "stream_ended"})
                break

            if frame_id % stride == 0:
                loop = asyncio.get_event_loop()
                report = await loop.run_in_executor(
                    _executor, pipeline.process_frame, frame, frame_id
                )

                if report.has_violations:
                    await websocket.send_json(report.to_dict())

            frame_id += 1

            # Check for client messages (stop command)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                if msg == "stop":
                    break
            except asyncio.TimeoutError:
                pass

        cap.release()

    except WebSocketDisconnect:
        print("[Stream] Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

@router.post("/analyze/dataset", response_model=DatasetAnalysisResponse, tags=["Dataset"])
async def analyze_dataset(
    dataset_path: str = Query(..., description="Path to dataset directory"),
    dataset_type: str = Query("violation", description="'violation' or 'plate'"),
):
    """
    Validate and analyze a dataset for training readiness.

    Checks image-label alignment, class distribution, split ratios, and data quality.
    """
    from analysis.analyze_violation_dataset import analyze_violation_dataset
    from analysis.analyze_plate_dataset import analyze_plate_dataset

    if not Path(dataset_path).exists():
        raise HTTPException(status_code=404, detail=f"Dataset path not found: {dataset_path}")

    loop = asyncio.get_event_loop()

    if dataset_type == "violation":
        result = await loop.run_in_executor(_executor, analyze_violation_dataset, dataset_path)
    elif dataset_type == "plate":
        result = await loop.run_in_executor(_executor, analyze_plate_dataset, dataset_path)
    else:
        raise HTTPException(status_code=400, detail="dataset_type must be 'violation' or 'plate'")

    return DatasetAnalysisResponse(**result)


# ============================================================================
# SERVE ANNOTATED OUTPUT FILES
# ============================================================================

@router.get("/output/{filename}", tags=["Output"])
async def serve_output(filename: str):
    """Serve annotated output files (images/videos)."""
    filepath = config.OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(filepath))
