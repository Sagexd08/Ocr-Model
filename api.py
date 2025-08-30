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

# In-memory job and document storage
class JobStore:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.RLock()
    
    def add_job(self, job_id, filename):
        with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "filename": filename,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "progress": 0.0,
            }
    
    def update_job(self, job_id, **kwargs):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")
            
            for key, value in kwargs.items():
                self.jobs[job_id][key] = value
            
            self.jobs[job_id]["updated_at"] = datetime.now().isoformat()
    
    def get_job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                return None
            return self.jobs[job_id]
    
    def list_jobs(self, limit=10, offset=0):
        with self.lock:
            all_jobs = list(self.jobs.values())
            all_jobs.sort(key=lambda j: j["created_at"], reverse=True)
            return all_jobs[offset:offset+limit]

class DocumentStore:
    def __init__(self):
        self.documents = {}
        self.lock = threading.RLock()
    
    def add_document(self, job_id, document_data):
        with self.lock:
            self.documents[job_id] = document_data
    
    def get_document(self, job_id):
        with self.lock:
            if job_id not in self.documents:
                return None
            return self.documents[job_id]
    
    def update_document(self, job_id, **kwargs):
        with self.lock:
            if job_id not in self.documents:
                raise ValueError(f"Document {job_id} not found")
            
            for key, value in kwargs.items():
                self.documents[job_id][key] = value

# Initialize stores
job_store = JobStore()
document_store = DocumentStore()

# Pydantic models
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"

class DocumentType(str, Enum):
    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE = "image"
    DOCX = "docx"
    UNKNOWN = "unknown"

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: Optional[str] = None
    processing_time: Optional[float] = None

class TableData(BaseModel):
    id: int
    title: str
    rows: int
    cols: int
    data: List[List[Any]]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class Page(BaseModel):
    page_number: int
    text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    has_tables: bool = False
    tables: List[TableData] = []

class DocumentMetadata(BaseModel):
    page_count: int
    document_type: DocumentType
    total_tables: int
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time: float
    file_size: int
    creation_date: Optional[str] = None
    last_modified_date: Optional[str] = None
    author: Optional[str] = None
    keywords: List[str] = []

class DocumentData(BaseModel):
    job_id: str
    filename: str
    pages: List[Page]
    metadata: DocumentMetadata

class ReviewData(BaseModel):
    job_id: str
    corrected_content: Dict[str, Any]
    reviewer_notes: Optional[str] = None

class JobListResponse(BaseModel):
    jobs: List[Dict[str, Any]]
    total: int

# Helper functions
def is_valid_file_type(filename):
    """Check if file type is supported."""
    allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.docx']
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed_extensions

def get_document_type(file_path):
    """Determine the document type based on file extension and content."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        # TODO: Add more sophisticated check for digital vs scanned PDF
        # For now, just assume it's digital
        return DocumentType.DIGITAL_PDF
    elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
        return DocumentType.IMAGE
    elif ext == '.docx':
        return DocumentType.DOCX
    else:
        return DocumentType.UNKNOWN

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    try:
        # This is a simplified implementation
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use pytesseract for OCR
        text = pytesseract.image_to_string(gray)
        
        # For demonstration, generate a confidence score
        confidence = 0.85  # In a real implementation, this would come from the OCR engine
        
        return text, confidence
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return "", 0.0

def detect_tables_in_image(image_path):
    """Detect tables in an image."""
    # This is a simplified implementation
    # In a real implementation, we would use a table detection model
    
    # For demonstration, return a dummy table
    table = {
        "id": 1,
        "title": "Detected Table",
        "rows": 3,
        "cols": 3,
        "data": [
            ["Header 1", "Header 2", "Header 3"],
            ["Data 1", "Data 2", "Data 3"],
            ["Data 4", "Data 5", "Data 6"]
        ],
        "confidence": 0.80
    }
    
    return [table]

def process_pdf(pdf_path):
    """Process a PDF document."""
    # This is a simplified implementation
    # In a real implementation, we would use more sophisticated PDF processing
    
    pages = []
    try:
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path)
        
        for i, img in enumerate(images):
            # Save the image temporarily
            img_path = str(TEMP_PATH / f"temp_page_{i}.png")
            img.save(img_path)
            
            # Extract text and detect tables
            text, confidence = extract_text_from_image(img_path)
            tables = detect_tables_in_image(img_path)
            
            pages.append({
                "page_number": i + 1,
                "text": text,
                "confidence": confidence,
                "has_tables": len(tables) > 0,
                "tables": tables
            })
            
            # Clean up temporary image
            os.remove(img_path)
            
        return pages
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return []

def store_uploaded_file(file, job_id):
    """Store the uploaded file in the input directory."""
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".pdf"
    file_path = INPUT_PATH / f"{job_id}{file_extension}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path

async def process_document_async(job_id, file_path, confidence_threshold):
    """Process a document asynchronously."""
    try:
        # Update job status to processing
        job_store.update_job(job_id, status="processing", progress=0.1)
        
        # Start processing timer
        start_time = time.time()
        
        # Get file information
        file_size = os.path.getsize(file_path)
        filename = job_store.get_job(job_id)["filename"]
        document_type = get_document_type(file_path)
        
        # Process based on document type
        pages = []
        total_tables = 0
        overall_confidence = 0.0
        
        if document_type in [DocumentType.IMAGE]:
            # Process image
            job_store.update_job(job_id, progress=0.3)
            text, confidence = extract_text_from_image(file_path)
            tables = detect_tables_in_image(file_path)
            total_tables = len(tables)
            
            pages.append({
                "page_number": 1,
                "text": text,
                "confidence": confidence,
                "has_tables": total_tables > 0,
                "tables": tables
            })
            
            overall_confidence = confidence
        
        elif document_type in [DocumentType.DIGITAL_PDF, DocumentType.SCANNED_PDF]:
            # Process PDF
            job_store.update_job(job_id, progress=0.3)
            pages = process_pdf(file_path)
            
            # Calculate overall statistics
            confidences = [page["confidence"] for page in pages]
            total_tables = sum(len(page["tables"]) for page in pages)
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
        else:
            # Unsupported document type
            job_store.update_job(
                job_id,
                status="failed",
                progress=1.0,
                message=f"Unsupported document type: {document_type}"
            )
            return
        
        # Update progress
        job_store.update_job(job_id, progress=0.7)
        
        # Simulate some additional processing time
        await asyncio.sleep(1)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create document data
        document_data = {
            "job_id": job_id,
            "filename": filename,
            "pages": pages,
            "metadata": {
                "page_count": len(pages),
                "document_type": document_type,
                "total_tables": total_tables,
                "confidence_score": overall_confidence,
                "processing_time": processing_time,
                "file_size": file_size,
                "creation_date": datetime.now().isoformat(),
                "last_modified_date": datetime.now().isoformat(),
                "author": None,
                "keywords": []
            }
        }
        
        # Store document
        document_store.add_document(job_id, document_data)
        
        # Write to output file
        output_file = OUTPUT_PATH / f"{job_id}.json"
        with open(output_file, "w") as f:
            json.dump(document_data, f, indent=2)
        
        # Update job status
        job_store.update_job(
            job_id,
            status="completed",
            progress=1.0,
            completed_at=datetime.now().isoformat(),
            processing_duration=processing_time,
            confidence_score=overall_confidence
        )
        
        logger.info(f"Job {job_id} completed successfully in {processing_time:.2f} seconds")
    
    except Exception as e:
        # Handle errors
        error_msg = str(e)
        logger.error(f"Error processing job {job_id}: {error_msg}")
        job_store.update_job(
            job_id,
            status="failed",
            progress=1.0,
            error_message=error_msg
        )

# Create FastAPI app
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
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")
    
    if not is_valid_file_type(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported formats: PDF, JPEG, PNG, TIFF, DOCX"
        )
    
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create a job entry
    job_store.add_job(job_id, file.filename)
    
    # Store the file
    file_path = store_uploaded_file(file, job_id)
    
    # Process the document asynchronously
    background_tasks.add_task(
        process_document_async, job_id, file_path, confidence_threshold
    )
    
    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="pending",
        message="Document uploaded and queued for processing"
    )

@app.get("/api/v1/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a job.
    """
    job = job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        message=job.get("error_message"),
        processing_time=job.get("processing_duration")
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
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job {job_id} is not completed (status: {job['status']})"
        )
    
    # Check if document exists
    document = document_store.get_document(job_id)
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document for job {job_id} not found")
    
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
    
    # Update job status
    job_store.update_job(job_id, status="reviewed")
    
    # Store the corrected content
    review_file = OUTPUT_PATH / f"{job_id}_review.json"
    with open(review_file, "w") as f:
        json.dump(review_data.dict(), f, indent=2)
    
    return {"message": f"Review for job {job_id} submitted successfully"}

@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_jobs(limit: int = 10, offset: int = 0):
    """
    List all jobs with pagination.
    """
    jobs = job_store.list_jobs(limit=limit, offset=offset)
    return JobListResponse(jobs=jobs, total=len(jobs))

@app.get("/api/v1/download/{job_id}")
async def download_results(job_id: str, format: str = "json"):
    """
    Download the results of a job in different formats.
    """
    # Check if the job exists and is completed
    job = job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job {job_id} is not completed (status: {job['status']})"
        )
    
    # Check if document exists
    document = document_store.get_document(job_id)
    
    if not document:
        raise HTTPException(status_code=404, detail=f"Document for job {job_id} not found")
    
    # Prepare filename
    filename = job["filename"]
    base_filename = os.path.splitext(filename)[0]
    
    if format == "json":
        # Return JSON results
        output_file = OUTPUT_PATH / f"{job_id}.json"
        return FileResponse(
            path=output_file,
            filename=f"{base_filename}_results.json",
            media_type="application/json"
        )
    
    elif format == "csv":
        # Convert results to CSV
        # This is a simplified implementation that extracts tables
        csv_file = TEMP_PATH / f"{job_id}_results.csv"
        
        # Extract all tables from the document
        all_tables = []
        for page in document["pages"]:
            for table in page["tables"]:
                all_tables.append({
                    "page": page["page_number"],
                    "table_id": table["id"],
                    "title": table["title"],
                    "data": table["data"]
                })
        
        # If there are tables, save them to CSV
        if all_tables:
            with open(csv_file, "w", newline="") as f:
                f.write("Page,TableID,Title,Data\n")
                for table in all_tables:
                    data_str = json.dumps(table["data"])
                    f.write(f"{table['page']},{table['table_id']},{table['title']},{data_str}\n")
            
            return FileResponse(
                path=csv_file,
                filename=f"{base_filename}_results.csv",
                media_type="text/csv"
            )
        else:
            raise HTTPException(status_code=400, detail="No tables found in the document")
    
    elif format == "txt":
        # Return plain text results
        txt_file = TEMP_PATH / f"{job_id}_results.txt"
        
        with open(txt_file, "w") as f:
            f.write(f"Document: {filename}\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Processing Time: {document['metadata']['processing_time']} seconds\n")
            f.write(f"Confidence Score: {document['metadata']['confidence_score']}\n")
            f.write("\n--- CONTENT ---\n\n")
            
            for page in document["pages"]:
                f.write(f"--- Page {page['page_number']} ---\n")
                f.write(page["text"])
                f.write("\n\n")
        
        return FileResponse(
            path=txt_file,
            filename=f"{base_filename}_results.txt",
            media_type="text/plain"
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "CurioScan OCR API", 
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            "/api/v1/upload",
            "/api/v1/status/{job_id}",
            "/api/v1/results/{job_id}",
            "/api/v1/review/{job_id}",
            "/api/v1/jobs",
            "/api/v1/download/{job_id}",
            "/api/v1/health"
        ]
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting CurioScan API server...")
    # Any additional initialization can go here

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down CurioScan API server...")
    # Any cleanup can go here

if __name__ == "__main__":
    logger.info("Starting CurioScan API server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
