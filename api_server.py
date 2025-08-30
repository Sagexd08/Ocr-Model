"""
Advanced API server for the CurioScan OCR project.
This version implements more realistic features and workflows.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import uuid
import os
import io
import shutil
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import cv2
import pytesseract
import pdf2image
import pandas as pd
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("curioscan-api")

# Storage paths
STORAGE_PATH = Path("./data/storage")
INPUT_PATH = STORAGE_PATH / "input"
OUTPUT_PATH = STORAGE_PATH / "output"
TEMP_PATH = STORAGE_PATH / "temp"
MODELS_PATH = Path("./models")

# Create directories if they don't exist
for path in [INPUT_PATH, OUTPUT_PATH, TEMP_PATH, MODELS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Create FastAPI app at module level
app = FastAPI(
    title="CurioScan API",
    description="Advanced API for CurioScan OCR document processing",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job and document storage
class JobStore:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        
    def add_job(self, job_id, job_data):
        with self.lock:
            self.jobs[job_id] = job_data
            
    def get_job(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job(self, job_id, updates):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)
                return True
            return False
    
    def get_all_jobs(self):
        with self.lock:
            return self.jobs.copy()

class DocumentStore:
    def __init__(self):
        self.documents = {}
        self.lock = threading.Lock()
        
    def add_document(self, job_id, document_data):
        with self.lock:
            self.documents[job_id] = document_data
            
    def get_document(self, job_id):
        with self.lock:
            return self.documents.get(job_id)
    
    def update_document(self, job_id, updates):
        with self.lock:
            if job_id in self.documents:
                self.documents[job_id].update(updates)
                return True
            return False

# Create global stores
job_store = JobStore()
document_store = DocumentStore()

# Document and table renderers
class RendererType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORM = "form"
    CHART = "chart"
    UNKNOWN = "unknown"

# Document processing status
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"

# Model and request schemas
class UploadRequest(BaseModel):
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    document_type: Optional[str] = None
    callback_url: Optional[str] = None

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_duration: Optional[float] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None

class TableData(BaseModel):
    id: int
    title: Optional[str] = None
    rows: int
    cols: int
    data: List[List[Any]]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    location: Dict[str, float] = Field(default_factory=dict)
    page_number: int = 1

class DocumentMetadata(BaseModel):
    page_count: int
    confidence_score: float
    processing_time: float
    document_type: str = "unknown"
    language: Optional[str] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    page_dimensions: Optional[Dict[str, List[int]]] = None
    
class PageData(BaseModel):
    page_number: int
    text_content: str
    confidence: float
    word_count: Optional[int] = None
    tables: List[int] = Field(default_factory=list)
    images: List[int] = Field(default_factory=list)
    
class DocumentData(BaseModel):
    job_id: str
    filename: str
    text_content: str
    pages: List[PageData]
    tables: List[TableData]
    metadata: DocumentMetadata

class ReviewData(BaseModel):
    job_id: str
    corrected_content: Dict[str, Any]
    feedback: Optional[str] = None
    
class RetrainRequest(BaseModel):
    job_ids: List[str]
    model_name: str
    parameters: Optional[Dict[str, Any]] = None

# Image processing utilities
def read_image(file_path):
    """Read image from file path."""
    try:
        if isinstance(file_path, (str, Path)):
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Failed to load image: {file_path}")
            return img
        return file_path
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise

def preprocess_image(image):
    """Preprocess image for OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        return thresh
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return image

def detect_tables(image):
    """Detect tables in an image."""
    try:
        # This is a simplified table detection
        # In a real system, you'd use a more sophisticated approach
        preprocessed = preprocess_image(image)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            preprocessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours to find potential tables
        tables = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter small contours
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Tables often have a reasonable aspect ratio
            if 0.5 <= aspect_ratio <= 2.0:
                tables.append({
                    "id": i,
                    "bbox": [x, y, w, h],
                    "area": area
                })
        
        return tables
    except Exception as e:
        logger.error(f"Error detecting tables: {e}")
        return []

def extract_text_from_image(image, lang='eng'):
    """Extract text from an image using OCR."""
    try:
        preprocessed = preprocess_image(image)
        text = pytesseract.image_to_string(preprocessed, lang=lang)
        data = pytesseract.image_to_data(preprocessed, lang=lang, output_type=pytesseract.Output.DICT)
        
        # Calculate confidence score
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "text": text,
            "confidence": avg_confidence / 100.0,  # Convert to 0-1 range
            "word_count": len([word for word in data['text'] if word.strip()]),
            "char_count": sum(len(word) for word in data['text'] if word.strip())
        }
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return {
            "text": "",
            "confidence": 0.0,
            "word_count": 0,
            "char_count": 0
        }

def extract_tables_from_image(image):
    """Extract tables from an image."""
    tables = detect_tables(image)
    result = []
    
    for i, table_info in enumerate(tables):
        x, y, w, h = table_info["bbox"]
        table_img = image[y:y+h, x:x+w]
        
        # Extract text using OCR
        text_data = extract_text_from_image(table_img)
        
        # In a real system, you'd use a more sophisticated table extraction
        # Here we're just creating a simple grid
        rows = 3
        cols = 3
        data = [["" for _ in range(cols)] for _ in range(rows)]
        
        # Fill with some example data
        data[0] = [f"Col {j+1}" for j in range(cols)]
        for r in range(1, rows):
            for c in range(cols):
                data[r][c] = f"Cell {r},{c}"
        
        result.append({
            "id": i + 1,
            "title": f"Table {i + 1}",
            "rows": rows,
            "cols": cols,
            "data": data,
            "confidence": text_data["confidence"],
            "location": {
                "x": x / image.shape[1],  # Normalize to 0-1 range
                "y": y / image.shape[0],
                "width": w / image.shape[1],
                "height": h / image.shape[0]
            },
            "page_number": 1  # Default to first page
        })
    
    return result

def process_pdf(file_path):
    """Process a PDF document."""
    try:
        # Convert PDF to images
        images = pdf2image.convert_from_path(file_path)
        result = {
            "pages": [],
            "tables": [],
            "metadata": {
                "page_count": len(images),
                "confidence_score": 0.0,
                "processing_time": 0.0,
                "document_type": "pdf",
                "language": "eng",
                "word_count": 0,
                "character_count": 0,
                "page_dimensions": {}
            }
        }
        
        total_confidence = 0.0
        total_word_count = 0
        total_char_count = 0
        
        for i, img in enumerate(images):
            # Convert PIL image to cv2 format
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Extract text
            ocr_result = extract_text_from_image(cv_img)
            text_content = ocr_result["text"]
            confidence = ocr_result["confidence"]
            word_count = ocr_result["word_count"]
            char_count = ocr_result["char_count"]
            
            # Update totals
            total_confidence += confidence
            total_word_count += word_count
            total_char_count += char_count
            
            # Extract tables
            tables = extract_tables_from_image(cv_img)
            for table in tables:
                table["page_number"] = i + 1
                result["tables"].append(table)
                
            # Add page data
            result["pages"].append({
                "page_number": i + 1,
                "text_content": text_content,
                "confidence": confidence,
                "word_count": word_count,
                "tables": [t["id"] for t in tables],
                "images": []
            })
            
            # Store page dimensions
            result["metadata"]["page_dimensions"][f"page_{i+1}"] = list(img.size)
            
        # Calculate average confidence
        avg_confidence = total_confidence / len(images) if images else 0.0
        result["metadata"]["confidence_score"] = avg_confidence
        result["metadata"]["word_count"] = total_word_count
        result["metadata"]["character_count"] = total_char_count
        
        return result
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

def process_image_file(file_path):
    """Process an image file."""
    try:
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        # Extract text
        ocr_result = extract_text_from_image(img)
        text_content = ocr_result["text"]
        confidence = ocr_result["confidence"]
        word_count = ocr_result["word_count"]
        char_count = ocr_result["char_count"]
        
        # Extract tables
        tables = extract_tables_from_image(img)
        
        result = {
            "pages": [{
                "page_number": 1,
                "text_content": text_content,
                "confidence": confidence,
                "word_count": word_count,
                "tables": [t["id"] for t in tables],
                "images": []
            }],
            "tables": tables,
            "metadata": {
                "page_count": 1,
                "confidence_score": confidence,
                "processing_time": 0.0,
                "document_type": "image",
                "language": "eng",
                "word_count": word_count,
                "character_count": char_count,
                "page_dimensions": {
                    "page_1": [img.shape[1], img.shape[0]]
                }
            }
        }
        
        return result
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

# File upload utility
def store_uploaded_file(file, job_id):
    """Store the uploaded file in the input directory."""
    try:
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ".pdf"
        file_path = INPUT_PATH / f"{job_id}{file_extension}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return file_path
    except Exception as e:
        logger.error(f"Error storing file: {e}")
        raise

# Document processing function
async def process_document(job_id, file_path, confidence_threshold):
    """Process the document with OCR."""
    try:
        logger.info(f"Processing job {job_id} with file {file_path}")
        
        # Update job status to processing
        job_store.update_job(job_id, {
            "status": JobStatus.PROCESSING,
            "updated_at": datetime.now().isoformat()
        })
        
        # Record start time
        start_time = time.time()
        
        # Get file information
        job = job_store.get_job(job_id)
        filename = job["filename"]
        file_extension = os.path.splitext(str(file_path))[1].lower()
        
        # Process based on file type
        if file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            result = process_image_file(file_path)
            document_type = "scanned_image"
        elif file_extension in ['.pdf']:
            result = process_pdf(file_path)
            document_type = "pdf"
        else:
            # Default to simple text extraction for unknown types
            result = {
                "pages": [{
                    "page_number": 1,
                    "text_content": f"Unsupported file type: {file_extension}",
                    "confidence": 0.0,
                    "word_count": 0,
                    "tables": [],
                    "images": []
                }],
                "tables": [],
                "metadata": {
                    "page_count": 1,
                    "confidence_score": 0.0,
                    "processing_time": 0.0,
                    "document_type": "unknown",
                    "language": "eng",
                    "word_count": 0,
                    "character_count": 0,
                    "page_dimensions": {}
                }
            }
        
        # Calculate processing time
        processing_time = time.time() - start_time
        result["metadata"]["processing_time"] = processing_time
        result["metadata"]["document_type"] = document_type
        
        # Combine text from all pages
        text_content = "\n".join([page["text_content"] for page in result["pages"]])
        
        # Create document data
        document_data = {
            "job_id": job_id,
            "filename": filename,
            "text_content": text_content,
            "pages": result["pages"],
            "tables": result["tables"],
            "metadata": result["metadata"]
        }
        
        # Store document
        document_store.add_document(job_id, document_data)
        
        # Write to output file
        output_file = OUTPUT_PATH / f"{job_id}.json"
        with open(output_file, "w") as f:
            # Convert to dict for JSON serialization
            json_data = {
                "job_id": job_id,
                "filename": filename,
                "text_content": text_content,
                "pages": [page for page in result["pages"]],
                "tables": [table for table in result["tables"]],
                "metadata": {k: v for k, v in result["metadata"].items()}
            }
            json.dump(json_data, f, indent=2)
        
        # Update job status
        job_store.update_job(job_id, {
            "status": JobStatus.COMPLETED,
            "updated_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "processing_duration": processing_time,
            "confidence_score": result["metadata"]["confidence_score"]
        })
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing job {job_id}: {error_msg}")
        
        # Update job status to failed
        job_store.update_job(job_id, {
            "status": JobStatus.FAILED,
            "updated_at": datetime.now().isoformat(),
            "error_message": error_msg
        })
        
        # Return error status
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            progress=1.0,
            created_at=job_store.get_job(job_id)["created_at"],
            updated_at=datetime.now().isoformat(),
            error_message=error_msg
        )


# Routes
@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.8, ge=0.0, le=1.0),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a document for OCR processing.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Get filename or use a default
        filename = file.filename or "unknown.pdf"
        
        # Create a job entry
        job_data = {
            "job_id": job_id,
            "filename": filename,
            "status": JobStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        job_store.add_job(job_id, job_data)
        
        # Store the file
        file_path = store_uploaded_file(file, job_id)
        
        # Process the document in the background
        background_tasks.add_task(
            process_document, job_id, file_path, confidence_threshold
        )
        
        logger.info(f"Document {filename} uploaded with job ID {job_id}")
        
        return UploadResponse(
            job_id=job_id,
            filename=filename,
            status=JobStatus.PENDING,
            message="Document uploaded and queued for processing"
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error uploading document: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/v1/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a job.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Calculate progress based on status
    progress = 0.0
    if job["status"] == JobStatus.PENDING:
        progress = 0.0
    elif job["status"] == JobStatus.PROCESSING:
        progress = 0.5  # Simplified progress indication
    elif job["status"] == JobStatus.COMPLETED:
        progress = 1.0
    elif job["status"] == JobStatus.FAILED:
        progress = 1.0  # Failed but processing is complete
    elif job["status"] == JobStatus.REVIEWED:
        progress = 1.0
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=progress,
        created_at=job["created_at"],
        updated_at=job.get("updated_at"),
        completed_at=job.get("completed_at"),
        processing_duration=job.get("processing_duration"),
        confidence_score=job.get("confidence_score"),
        error_message=job.get("error_message")
    )

@app.get("/api/v1/results/{job_id}", response_model=DocumentData)
async def get_job_results(job_id: str):
    """
    Get the results of a job.
    """
    # Check if the job exists
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check if job is completed
    if job["status"] != JobStatus.COMPLETED and job["status"] != JobStatus.REVIEWED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job {job_id} is not completed (status: {job['status']})"
        )
    
    # Check if document exists
    document = document_store.get_document(job_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document for job {job_id} not found")
    
    # Return document data
    return document

@app.post("/api/v1/review/{job_id}")
async def submit_review(job_id: str, review_data: ReviewData):
    """
    Submit a review with corrected data for a document.
    """
    # Check if the job exists
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check if document exists
    document = document_store.get_document(job_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document for job {job_id} not found")
    
    try:
        # Update job status
        job_store.update_job(job_id, {
            "status": JobStatus.REVIEWED,
            "updated_at": datetime.now().isoformat()
        })
        
        # Store the corrected content
        review_file = OUTPUT_PATH / f"{job_id}_review.json"
        with open(review_file, "w") as f:
            json.dump(review_data.dict(), f, indent=2)
        
        logger.info(f"Review submitted for job {job_id}")
        
        return {"message": f"Review for job {job_id} submitted successfully"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error submitting review for job {job_id}: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v1/retrain")
async def retrain_model(retrain_request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Retrain a model based on reviewed documents.
    """
    try:
        # Validate job IDs
        valid_job_ids = []
        for job_id in retrain_request.job_ids:
            job = job_store.get_job(job_id)
            if job and job["status"] == JobStatus.REVIEWED:
                valid_job_ids.append(job_id)
        
        if not valid_job_ids:
            raise HTTPException(
                status_code=400, 
                detail="No valid reviewed jobs found for retraining"
            )
        
        # In a real implementation, you would initiate model retraining here
        # This is a mock implementation
        
        logger.info(f"Initiating retraining of model {retrain_request.model_name} with {len(valid_job_ids)} jobs")
        
        # Simulate background processing
        await asyncio.sleep(0.1)
        
        return {
            "message": f"Retraining of model {retrain_request.model_name} initiated with {len(valid_job_ids)} jobs",
            "status": "pending",
            "job_count": len(valid_job_ids)
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error initiating model retraining: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "job_count": len(job_store.get_all_jobs())
    }

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "CurioScan OCR API", 
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/upload",
            "/api/v1/status/{job_id}",
            "/api/v1/results/{job_id}",
            "/api/v1/review/{job_id}",
            "/api/v1/retrain",
            "/api/v1/health"
        ],
        "docs_url": "/docs"
    }

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down CurioScan API server...")
    # Any cleanup can go here

if __name__ == "__main__":
    logger.info("Starting CurioScan API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
