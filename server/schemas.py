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
    detections: List[Dict[str, Any]] = Field(default_factory=list)
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


# ── NEW: Dashboard Data Models ──

class ViolationRecord(BaseModel):
    """A persisted violation record from the database."""
    id: int
    plate_number: str
    violation_type: str
    confidence: float
    severity: str
    location: str
    fine_amount: int
    status: str
    image_path: Optional[str] = None
    created_at: Optional[str] = None


class ViolationListResponse(BaseModel):
    """Paginated list of violations."""
    success: bool = True
    violations: List[ViolationRecord]
    total: int
    page: int
    limit: int


class HourlyCount(BaseModel):
    hour: str
    count: int

class TypeCount(BaseModel):
    name: str
    value: int

class ViolationStatsResponse(BaseModel):
    """Aggregated violation statistics for the dashboard."""
    success: bool = True
    total_today: int = 0
    total_week: int = 0
    total_month: int = 0
    challans_issued: int = 0
    by_type: List[TypeCount] = Field(default_factory=list)
    by_hour: List[HourlyCount] = Field(default_factory=list)


class EntityResponse(BaseModel):
    """All violation data for a specific license plate."""
    success: bool = True
    plate_number: str
    total_violations: int = 0
    total_fines: int = 0
    unpaid_fines: int = 0
    violations: List[ViolationRecord] = Field(default_factory=list)