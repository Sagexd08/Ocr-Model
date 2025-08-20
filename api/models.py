"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RenderType(str, Enum):
    """Document render type classification."""
    DIGITAL_PDF = "digital_pdf"
    SCANNED_IMAGE = "scanned_image"
    PHOTOGRAPH = "photograph"
    DOCX = "docx"
    FORM = "form"
    TABLE_HEAVY = "table_heavy"
    INVOICE = "invoice"
    HANDWRITTEN = "handwritten"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    
    @validator('x2')
    def x2_greater_than_x1(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @validator('y2')
    def y2_greater_than_y1(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v


class Provenance(BaseModel):
    """Provenance information for extracted data."""
    file: str = Field(..., description="Source file name")
    page: int = Field(..., description="Page number (1-indexed)")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    token_ids: List[int] = Field(..., description="OCR token IDs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")


class ExtractedRow(BaseModel):
    """Single row of extracted data with provenance."""
    row_id: str = Field(..., description="Unique row identifier")
    page: int = Field(..., description="Page number")
    region_id: str = Field(..., description="Region identifier")
    bbox: List[float] = Field(..., description="Row bounding box [x1, y1, x2, y2]")
    columns: Dict[str, Any] = Field(..., description="Column data")
    provenance: Provenance = Field(..., description="Data provenance")
    needs_review: bool = Field(default=False, description="Requires human review")


class ExtractionResult(BaseModel):
    """Complete extraction result."""
    rows: List[ExtractedRow] = Field(..., description="Extracted rows")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_time: float = Field(..., description="Total processing time in seconds")
    render_type: RenderType = Field(..., description="Detected render type")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")


class UploadResponse(BaseModel):
    """Response for file upload."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Initial job status")
    message: str = Field(default="File uploaded successfully", description="Status message")


class StatusResponse(BaseModel):
    """Response for job status check."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress percentage")
    message: Optional[str] = Field(None, description="Status message")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    preview: Optional[Dict[str, Any]] = Field(None, description="Preview data")


class JobResponse(BaseModel):
    """Detailed job information."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="File MIME type")
    render_type: Optional[RenderType] = Field(None, description="Detected render type")
    progress: float = Field(default=0.0, description="Progress percentage")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[ExtractionResult] = Field(None, description="Extraction result")


class WebhookRequest(BaseModel):
    """Webhook registration request."""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Webhook secret for verification")


class WebhookResponse(BaseModel):
    """Webhook registration response."""
    webhook_id: str = Field(..., description="Webhook identifier")
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Subscribed events")
    created_at: datetime = Field(..., description="Registration time")


class ReviewItem(BaseModel):
    """Item requiring human review."""
    item_id: str = Field(..., description="Item identifier")
    job_id: str = Field(..., description="Parent job identifier")
    page: int = Field(..., description="Page number")
    bbox: List[float] = Field(..., description="Bounding box")
    original_text: str = Field(..., description="Original OCR text")
    confidence: float = Field(..., description="OCR confidence")
    suggested_text: Optional[str] = Field(None, description="Suggested correction")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ReviewResponse(BaseModel):
    """Review items response."""
    items: List[ReviewItem] = Field(..., description="Items requiring review")
    total_count: int = Field(..., description="Total number of review items")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")


class ReviewUpdate(BaseModel):
    """Review item update."""
    item_id: str = Field(..., description="Item identifier")
    action: str = Field(..., description="Action: accept, reject, edit")
    corrected_text: Optional[str] = Field(None, description="Corrected text if action is edit")


class RetrainRequest(BaseModel):
    """Model retraining request."""
    model_type: str = Field(..., description="Model type to retrain")
    dataset_path: Optional[str] = Field(None, description="Custom dataset path")
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Config overrides")
    dry_run: bool = Field(default=False, description="Dry run mode")


class RetrainResponse(BaseModel):
    """Model retraining response."""
    retrain_job_id: str = Field(..., description="Retraining job identifier")
    status: str = Field(..., description="Retraining status")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    message: str = Field(..., description="Status message")
