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
    filename = Column(String)
    status = Column(String, default=JobStatus.PENDING.value)
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
