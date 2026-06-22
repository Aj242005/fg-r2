"""
Database Models — SQLAlchemy ORM
=================================
Tables for persisting violations detected by the AI pipeline.
"""

import datetime
from sqlalchemy import String, Float, Integer, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from server.database import Base


class Violation(Base):
    """A single traffic violation detected and logged by the system."""
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_number: Mapped[str] = mapped_column(String(20), nullable=True, index=True)
    violation_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False, default="MEDIUM")
    location: Mapped[str] = mapped_column(String(200), nullable=True, default="Camera Feed")
    fine_amount: Mapped[int] = mapped_column(Integer, nullable=False, default=1000)
    status: Mapped[str] = mapped_column(String(10), nullable=False, default="Unpaid", index=True)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    def to_dict(self):
        return {
            "id": self.id,
            "plate_number": self.plate_number or "UNKNOWN",
            "violation_type": self.violation_type,
            "confidence": self.confidence,
            "severity": self.severity,
            "location": self.location,
            "fine_amount": self.fine_amount,
            "status": self.status,
            "image_path": self.image_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
