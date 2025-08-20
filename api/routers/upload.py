"""
File upload endpoint for CurioScan API.
"""

import uuid
import os
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db, Job
from ..models import UploadResponse, JobStatus
from ..dependencies import get_current_user, rate_limit, validate_file
from ..storage import upload_file_to_storage
from ..tasks import process_document_task

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)
settings = get_settings()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = None,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user),
    _: None = Depends(rate_limit)
):
    """
    Upload a file for OCR processing.
    
    Supported file types:
    - PDF (digital and scanned)
    - Images (JPEG, PNG, TIFF)
    - DOCX documents
    
    Returns a job ID for tracking processing status.
    """
    try:
        # Validate file
        await validate_file(file)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        logger.info(f"Starting upload for job {job_id}, file: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Upload to storage
        input_path = await upload_file_to_storage(
            file_content, 
            f"input/{job_id}/{file.filename}"
        )
        
        # Create job record
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            file_name=file.filename,
            file_size=file_size,
            mime_type=file.content_type,
            input_path=input_path,
            processing_metadata={
                "confidence_threshold": confidence_threshold or settings.default_confidence_threshold,
                "uploaded_by": current_user.get("user_id") if current_user else "anonymous"
            }
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Queue processing task
        background_tasks.add_task(
            queue_processing_task,
            job_id,
            input_path,
            confidence_threshold or settings.default_confidence_threshold
        )
        
        logger.info(f"File uploaded successfully for job {job_id}")
        
        return UploadResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="File uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


async def queue_processing_task(
    job_id: str, 
    input_path: str, 
    confidence_threshold: float
):
    """Queue the document processing task."""
    try:
        # Import here to avoid circular imports
        from ..celery_app import celery_app
        
        # Queue the processing task
        task = celery_app.send_task(
            'worker.tasks.process_document',
            args=[job_id, input_path, confidence_threshold],
            queue='default'
        )
        
        logger.info(f"Queued processing task {task.id} for job {job_id}")
        
    except Exception as e:
        logger.error(f"Failed to queue processing task for job {job_id}: {str(e)}")
        # Update job status to failed
        from ..database import SessionLocal
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.error_message = f"Failed to queue processing: {str(e)}"
                db.commit()
        finally:
            db.close()


@router.post("/upload/batch")
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    confidence_threshold: Optional[float] = None,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user),
    _: None = Depends(rate_limit)
):
    """
    Upload multiple files for batch processing.
    
    Returns a list of job IDs for tracking processing status.
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 10 files"
        )
    
    job_ids = []
    
    for file in files:
        try:
            # Process each file individually
            result = await upload_file(
                background_tasks=background_tasks,
                file=file,
                confidence_threshold=confidence_threshold,
                db=db,
                current_user=current_user
            )
            job_ids.append(result.job_id)
            
        except Exception as e:
            logger.error(f"Failed to upload file {file.filename}: {str(e)}")
            # Continue with other files
            continue
    
    return {
        "job_ids": job_ids,
        "total_files": len(files),
        "successful_uploads": len(job_ids),
        "failed_uploads": len(files) - len(job_ids)
    }
