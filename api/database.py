"""
Database models and connection management for CurioScan API.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, 
    Boolean, Text, JSON, LargeBinary, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
import sqlalchemy as sa

from .config import get_settings

settings = get_settings()

# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    """Job table for tracking processing jobs."""
    __tablename__ = "jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(255), unique=True, index=True, nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    render_type = Column(String(50), nullable=True)
    progress = Column(Float, default=0.0)
    confidence_score = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Storage paths
    input_path = Column(String(500), nullable=False)
    output_path = Column(String(500), nullable=True)
    
    # Processing metadata
    processing_metadata = Column(JSON, nullable=True)
    
    # Relationships
    extraction_results = relationship("ExtractionResult", back_populates="job", cascade="all, delete-orphan")
    review_items = relationship("ReviewItem", back_populates="job", cascade="all, delete-orphan")


class ExtractionResult(Base):
    """Table for storing extraction results."""
    __tablename__ = "extraction_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    row_id = Column(String(255), nullable=False)
    page = Column(Integer, nullable=False)
    region_id = Column(String(255), nullable=False)
    
    # Bounding box
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    # Extracted data
    columns_data = Column(JSON, nullable=False)
    
    # Provenance
    source_file = Column(String(255), nullable=False)
    source_page = Column(Integer, nullable=False)
    source_bbox_x1 = Column(Float, nullable=False)
    source_bbox_y1 = Column(Float, nullable=False)
    source_bbox_x2 = Column(Float, nullable=False)
    source_bbox_y2 = Column(Float, nullable=False)
    token_ids = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Review status
    needs_review = Column(Boolean, default=False)
    reviewed = Column(Boolean, default=False)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    job = relationship("Job", back_populates="extraction_results")


class ReviewItem(Base):
    """Table for items requiring human review."""
    __tablename__ = "review_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(String(255), unique=True, index=True, nullable=False)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    
    # Location
    page = Column(Integer, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    # OCR data
    original_text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    suggested_text = Column(Text, nullable=True)
    
    # Review status
    status = Column(String(50), default="pending")  # pending, accepted, rejected, edited
    corrected_text = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String(255), nullable=True)
    
    # Context
    context_data = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    job = relationship("Job", back_populates="review_items")


class Webhook(Base):
    """Table for webhook registrations."""
    __tablename__ = "webhooks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(String(255), unique=True, index=True, nullable=False)
    url = Column(String(500), nullable=False)
    events = Column(JSON, nullable=False)
    secret = Column(String(255), nullable=True)
    active = Column(Boolean, default=True)
    
    # Statistics
    total_calls = Column(Integer, default=0)
    successful_calls = Column(Integer, default=0)
    failed_calls = Column(Integer, default=0)
    last_called_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class ModelCheckpoint(Base):
    """Table for tracking model checkpoints and versions."""
    __tablename__ = "model_checkpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(255), nullable=False)
    version = Column(String(100), nullable=False)
    checkpoint_path = Column(String(500), nullable=False)
    
    # Metrics
    metrics = Column(JSON, nullable=True)
    
    # Training info
    training_config = Column(JSON, nullable=True)
    training_dataset = Column(String(255), nullable=True)
    training_duration = Column(Float, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# Database dependency
def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
