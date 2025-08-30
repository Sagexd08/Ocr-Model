from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime, Float, Boolean
from sqlalchemy.sql import func
from .database import Base
import enum
from datetime import datetime
from pydantic import BaseModel

class JobStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    # Legacy field kept for compatibility
    filename = Column(String, nullable=True)
    # Preferred fields used by routers
    file_name = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String, nullable=True)
    status = Column(String, default=JobStatus.PENDING.value)
    progress = Column(Float, default=0.0)
    input_path = Column(String, nullable=True)
    processing_metadata = Column(JSON, nullable=True)
    render_type = Column(String, nullable=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime, nullable=True)
    processing_duration = Column(Float, nullable=True)  # in seconds
    error_message = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    webhook_sent = Column(Boolean, default=False)
    webhook_url = Column(String, nullable=True)

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    content = Column(JSON)

class CorrectedData(Base):
    __tablename__ = "corrected_data"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    corrected_content = Column(JSON)

# Pydantic models for API responses
class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str


# Minimal response models expected by routers/status.py
class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float | None = None
    message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    estimated_completion: datetime | None = None
    preview: dict | None = None

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    file_name: str | None = None
    file_size: int | None = None
    mime_type: str | None = None
    render_type: str | None = None
    progress: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    result: dict | None = None
