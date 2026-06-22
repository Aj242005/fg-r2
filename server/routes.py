"""
API Routes for Traffic Violation Detection Server
===================================================
Endpoints for image, video, stream analysis, health check, dataset validation,
and dashboard data (violations, entities, reports).
"""

import os
import time
import json
import asyncio
import tempfile
import uuid
import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.responses import FileResponse, JSONResponse, Response

from sqlalchemy import select, func, extract, case, and_
from sqlalchemy.ext.asyncio import AsyncSession

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
    ViolationListResponse,
    ViolationStatsResponse,
    EntityResponse,
    ViolationRecord,
)
from server.database import get_db
from server.db_models import Violation
from server.reports import generate_challan_pdf, generate_audit_report_pdf, FINE_MAP


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
# IMAGE ANALYSIS (now auto-saves violations to DB)
# ============================================================================

@router.post("/analyze/image", response_model=ImageAnalysisResponse, tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    return_annotated: bool = Query(True, description="Return annotated image"),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze a single image for traffic violations.
    Detected violations are automatically saved to the database.
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

    # ── Auto-save violations to database ──
    try:
        plates_found = [p.get("plate_text", "") for p in report_dict.get("plates", []) if p.get("plate_text")]
        primary_plate = plates_found[0] if plates_found else None

        for v in report_dict.get("violations", []):
            vtype = v.get("type", "unknown")
            fine = FINE_MAP.get(vtype, 1000)
            violation_row = Violation(
                plate_number=primary_plate,
                violation_type=vtype,
                confidence=v.get("confidence", 0),
                severity=v.get("severity", "MEDIUM"),
                location="Camera Feed",
                fine_amount=fine,
                status="Unpaid",
                image_path=annotated_url,
            )
            db.add(violation_row)
        await db.commit()
    except Exception as e:
        print(f"[DB] Warning: Could not save violations: {e}")
        await db.rollback()

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
    """
    pipeline = get_pipeline()

    # Save uploaded video to temp file
    suffix = Path(file.filename).suffix or ".mp4"
    temp_path = config.OUTPUT_DIR / f"upload_{uuid.uuid4().hex[:8]}{suffix}"

    with open(temp_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    try:
        loop = asyncio.get_event_loop()
        reports = await loop.run_in_executor(
            _executor,
            lambda: pipeline.process_video(
                str(temp_path), stride=stride, max_frames=max_frames
            ),
        )

        report_dicts = [r.to_dict() for r in reports]

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
        if temp_path.exists():
            temp_path.unlink()


# ============================================================================
# LIVE STREAM ANALYSIS (WebSocket)
# ============================================================================

@router.websocket("/analyze/stream")
async def analyze_stream(websocket: WebSocket):
    """WebSocket endpoint for live stream analysis."""
    await websocket.accept()
    pipeline = get_pipeline()

    try:
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
    """Validate and analyze a dataset for training readiness."""
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


# ============================================================================
# DASHBOARD: VIOLATIONS LIST
# ============================================================================

@router.get("/violations", response_model=ViolationListResponse, tags=["Dashboard"])
async def list_violations(
    violation_type: Optional[str] = Query(None, description="Filter by violation type"),
    status: Optional[str] = Query(None, description="Filter by status (Paid/Unpaid)"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all violations with optional filters and pagination."""
    query = select(Violation).order_by(Violation.created_at.desc())
    count_query = select(func.count(Violation.id))

    if violation_type:
        query = query.where(Violation.violation_type == violation_type)
        count_query = count_query.where(Violation.violation_type == violation_type)
    if status:
        query = query.where(Violation.status == status)
        count_query = count_query.where(Violation.status == status)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    query = query.offset((page - 1) * limit).limit(limit)
    result = await db.execute(query)
    rows = result.scalars().all()

    return ViolationListResponse(
        violations=[ViolationRecord(**r.to_dict()) for r in rows],
        total=total,
        page=page,
        limit=limit,
    )


# ============================================================================
# DASHBOARD: VIOLATION STATS (for charts)
# ============================================================================

@router.get("/violations/stats", response_model=ViolationStatsResponse, tags=["Dashboard"])
async def violation_stats(db: AsyncSession = Depends(get_db)):
    """Aggregated statistics for the Area Overview dashboard."""
    now = datetime.datetime.now(datetime.timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - datetime.timedelta(days=now.weekday())
    month_start = today_start.replace(day=1)

    # Total today
    r = await db.execute(select(func.count(Violation.id)).where(Violation.created_at >= today_start))
    total_today = r.scalar() or 0

    # Total this week
    r = await db.execute(select(func.count(Violation.id)).where(Violation.created_at >= week_start))
    total_week = r.scalar() or 0

    # Total this month
    r = await db.execute(select(func.count(Violation.id)).where(Violation.created_at >= month_start))
    total_month = r.scalar() or 0

    # Challans issued (all time)
    r = await db.execute(select(func.count(Violation.id)))
    challans_issued = r.scalar() or 0

    # By type (this month)
    r = await db.execute(
        select(Violation.violation_type, func.count(Violation.id))
        .where(Violation.created_at >= month_start)
        .group_by(Violation.violation_type)
    )
    by_type = [{"name": row[0], "value": row[1]} for row in r.all()]

    # By hour (today)
    r = await db.execute(
        select(
            extract("hour", Violation.created_at).label("hr"),
            func.count(Violation.id),
        )
        .where(Violation.created_at >= today_start)
        .group_by("hr")
        .order_by("hr")
    )
    by_hour = [{"hour": f"{int(row[0]):02d}:00", "count": row[1]} for row in r.all()]

    return ViolationStatsResponse(
        total_today=total_today,
        total_week=total_week,
        total_month=total_month,
        challans_issued=challans_issued,
        by_type=by_type,
        by_hour=by_hour,
    )


# ============================================================================
# DASHBOARD: ENTITY LOOKUP
# ============================================================================

@router.get("/entity/{plate_number}", response_model=EntityResponse, tags=["Dashboard"])
async def get_entity(plate_number: str, db: AsyncSession = Depends(get_db)):
    """Get all violations for a specific license plate."""
    query = (
        select(Violation)
        .where(Violation.plate_number == plate_number.upper())
        .order_by(Violation.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    violations = [ViolationRecord(**r.to_dict()) for r in rows]
    total_fines = sum(v.fine_amount for v in violations)
    unpaid = sum(v.fine_amount for v in violations if v.status == "Unpaid")

    return EntityResponse(
        plate_number=plate_number.upper(),
        total_violations=len(violations),
        total_fines=total_fines,
        unpaid_fines=unpaid,
        violations=violations,
    )


# ============================================================================
# PDF: E-CHALLAN DOWNLOAD
# ============================================================================

@router.get("/violations/{violation_id}/challan", tags=["Reports"])
async def download_challan(violation_id: int, db: AsyncSession = Depends(get_db)):
    """Generate and download an E-Challan PDF for a specific violation."""
    result = await db.execute(select(Violation).where(Violation.id == violation_id))
    violation = result.scalar_one_or_none()

    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")

    pdf_bytes = generate_challan_pdf(violation.to_dict())

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=E-Challan-{violation_id}.pdf"},
    )


# ============================================================================
# PDF: AUDIT REPORT DOWNLOAD
# ============================================================================

@router.get("/entity/{plate_number}/report", tags=["Reports"])
async def download_audit_report(plate_number: str, db: AsyncSession = Depends(get_db)):
    """Generate and download a full audit report PDF for a vehicle entity."""
    query = (
        select(Violation)
        .where(Violation.plate_number == plate_number.upper())
        .order_by(Violation.created_at.desc())
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No violations found for plate: {plate_number}")

    violations = [r.to_dict() for r in rows]
    pdf_bytes = generate_audit_report_pdf(plate_number.upper(), violations)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Audit-Report-{plate_number.upper()}.pdf"},
    )
