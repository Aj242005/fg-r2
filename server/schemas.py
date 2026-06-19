"""
Pydantic Schemas for API Request/Response Models
==================================================
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ── Response Models ──

class PlateInfo(BaseModel):
    """Plate detection result."""
    text: str = Field(..., description="Plate text from OCR")
    confidence: float = Field(..., ge=0, le=1)
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")


class ViolationDetail(BaseModel):
    """Single violation detected."""
    type: str = Field(..., description="Violation type identifier")
    confidence: float = Field(..., ge=0, le=1)
    severity: str = Field(..., description="HIGH, MEDIUM, or LOW")
    bbox: List[float] = Field(..., description="Bounding box [x1,y1,x2,y2]")
    plate: Optional[PlateInfo] = Field(None, description="Associated plate if found")
    details: Dict[str, Any] = Field(default_factory=dict)


class FrameReport(BaseModel):
    """Analysis result for a single frame."""
    frame_id: int
    timestamp: float
    violation_count: int
    plate_count: int
    violations: List[ViolationDetail]
    plates: List[Dict[str, Any]]
    detection_count: int
    processing_time_ms: float


class ImageAnalysisResponse(BaseModel):
    """Response for /analyze/image endpoint."""
    success: bool = True
    report: FrameReport
    annotated_image_url: Optional[str] = None


class VideoAnalysisResponse(BaseModel):
    """Response for /analyze/video endpoint."""
    success: bool = True
    total_frames_processed: int
    frames_with_violations: int
    reports: List[FrameReport]
    annotated_video_url: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)


class StreamStartRequest(BaseModel):
    """Request to start stream analysis."""
    stream_url: str = Field(..., description="RTSP or HTTP stream URL")
    stride: int = Field(default=5, ge=1, description="Process every Nth frame")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    violation_model_loaded: bool
    plate_model_loaded: bool
    violation_classes: int
    uptime_seconds: float


class DatasetAnalysisRequest(BaseModel):
    """Request for dataset analysis."""
    dataset_path: str = Field(..., description="Path to dataset directory")
    dataset_type: str = Field("violation", description="'violation' or 'plate'")


class DatasetAnalysisResponse(BaseModel):
    """Response for dataset analysis."""
    success: bool = True
    dataset_path: str
    dataset_type: str
    total_images: int = 0
    total_labels: int = 0
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    splits: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    is_valid: bool = True
