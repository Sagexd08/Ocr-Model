from typing import List, Dict, Any, Optional, Union, TypedDict
from pydantic import BaseModel, Field
import uuid
from enum import Enum, auto

class JobStatus(str, Enum):
    """Status of a processing job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentType(str, Enum):
    """Types of documents that can be processed"""
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"
    DOCX = "docx"
    UNKNOWN = "unknown"


class ProcessingMode(str, Enum):
    """Different processing modes for documents"""
    BASIC = "basic"          # Simple text extraction only
    STANDARD = "standard"    # Standard OCR with basic formatting
    ENHANCED = "enhanced"    # Enhanced OCR with layout preservation
    ADVANCED = "advanced"    # Full processing with tables, forms, and layout analysis
    CUSTOM = "custom"        # Custom processing with specific parameters


class ModelType(str, Enum):
    """Types of ML models used in the system"""
    OCR = "ocr"
    CLASSIFICATION = "classification"
    TABLE_DETECTION = "table_detection"
    LAYOUT_ANALYSIS = "layout_analysis"
    FORM_EXTRACTION = "form_extraction"


class StorageType(str, Enum):
    """Storage backend types"""
    LOCAL = "local"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"


class ExportFormat(str, Enum):
    """Supported export formats for results"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    TEXT = "text"
    XML = "xml"


class WebhookEventType(str, Enum):
    """Types of webhook events"""
    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"

class Bbox(BaseModel):
    """Bounding box coordinates normalized to [0,1] or in pixels"""
    x1: float
    y1: float
    x2: float
    y2: float

class Token(BaseModel):
    """Individual OCR token with text, position, and confidence"""
    id: str = Field(default_factory=lambda: f"tok_{uuid.uuid4().hex[:8]}")
    text: str
    bbox: Bbox
    confidence: float
    page_num: Optional[int] = None
    line_num: Optional[int] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

class Cell(BaseModel):
    """Table cell with text and token references"""
    text: str = ""
    tokens: List[Token] = Field(default_factory=list)
    normalized_text: Optional[str] = None
    confidence: float = 1.0
    needs_review: bool = False

class Region(BaseModel):
    """Document region like paragraph, heading, list, etc."""
    id: str = Field(default_factory=lambda: f"reg_{uuid.uuid4().hex[:8]}")
    type: str  # paragraph, heading, list, etc.
    bbox: Bbox
    text: str = ""
    normalized_text: Optional[str] = None
    tokens: List[Token] = Field(default_factory=list)
    confidence: float = 1.0
    attributes: Dict[str, Any] = Field(default_factory=dict)
    needs_review: bool = False

class Table(BaseModel):
    """Table with rows, columns and cells"""
    id: str = Field(default_factory=lambda: f"tbl_{uuid.uuid4().hex[:8]}")
    bbox: Bbox
    rows: List[List[Cell]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    needs_review: bool = False

class Page(BaseModel):
    """Document page with image, tokens, regions and tables"""
    page_num: int
    width: Optional[int] = None
    height: Optional[int] = None
    image: Any = None  # Can be a path to an image or a numpy array
    tokens: List[Token] = Field(default_factory=list)
    regions: List[Region] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    confidence: float = 1.0
    dpi: Optional[int] = None

class Document(BaseModel):
    """Full document with pages and metadata"""
    id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:8]}")
    pages: List[Page] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

class Row(BaseModel):
    """Output row with structured data and provenance"""
    row_id: str
    page: int
    region_id: str
    bbox: List[float]  # [x1, y1, x2, y2]
    columns: Dict[str, str]
    provenance: Dict[str, Any]
    needs_review: bool = False

class ProcessingResult(BaseModel):
    """Complete processing result with rows and metrics"""
    job_id: str
    document_id: str
    filename: str
    rows: List[Row] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

# Lightweight dict-based types for pipeline helpers
class OCRToken(TypedDict):
    text: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float

# Table data type definitions
RowTD = List[OCRToken]  # List of tokens in a row
TableContentTD = Dict[str, Any]  # {rows, columns}

class OCRPage(TypedDict):
    page_number: int
    tokens: List[OCRToken]
    page_bbox: List[int]

class TableContentTD(TypedDict):
    rows: List[List[OCRToken]]
    columns: List[str]