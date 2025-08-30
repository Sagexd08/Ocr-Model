"""
Job status endpoints for CurioScan API.
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..config import get_settings
from ..database import get_db
from ..models import Job, StatusResponse, JobResponse, JobStatus
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Get the current status of a processing job.
    
    Returns detailed status information including progress, estimated completion time,
    and preview data if available.
    """
    try:
        # Find the job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        # Calculate estimated completion time
        estimated_completion = None
        job_progress = getattr(job, 'progress', 0.0) or 0.0
        if job.status == JobStatus.PROCESSING.value and job_progress > 0:
            elapsed_time = (datetime.utcnow() - job.created_at).total_seconds()
            estimated_total_time = elapsed_time / job_progress
            remaining_time = max(0.0, estimated_total_time - elapsed_time)
            estimated_completion = datetime.utcnow() + timedelta(seconds=remaining_time)

        # Get preview data if available
        preview = None
        processing_metadata = getattr(job, 'processing_metadata', {}) or {}
        if isinstance(processing_metadata, dict) and "preview" in processing_metadata:
            preview = processing_metadata["preview"]

        return StatusResponse(
            job_id=job.job_id,
            status=JobStatus(job.status),
            progress=job_progress,
            message=_get_status_message(job),
            created_at=job.created_at,
            updated_at=job.updated_at,
            estimated_completion=estimated_completion,
            preview=preview
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    List processing jobs with optional filtering.
    
    Returns a paginated list of jobs with their current status and metadata.
    """
    try:
        query = db.query(Job)
        
        # Filter by status if provided
        if status:
            query = query.filter(Job.status == status.value)
        
        # Apply pagination
        jobs = query.order_by(desc(Job.created_at)).offset(offset).limit(limit).all()
        
        # Convert to response models
        job_responses = []
        for job in jobs:
            job_response = JobResponse(
                job_id=job.job_id,
                status=JobStatus(job.status),
                file_name=getattr(job, 'file_name', None) or getattr(job, 'filename', None),
                file_size=getattr(job, 'file_size', None),
                mime_type=getattr(job, 'mime_type', None),
                render_type=getattr(job, 'render_type', None),
                progress=getattr(job, 'progress', None),
                created_at=job.created_at,
                updated_at=job.updated_at,
                completed_at=job.completed_at,
                error_message=getattr(job, 'error_message', None)
            )
            job_responses.append(job_response)
        
        return job_responses
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_details(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Get detailed information about a specific job.
    
    Returns comprehensive job information including extraction results if completed.
    """
    try:
        # Find the job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        # Build response
        job_response = JobResponse(
            job_id=job.job_id,
            status=JobStatus(job.status),
            file_name=getattr(job, 'file_name', None) or getattr(job, 'filename', None),
            file_size=getattr(job, 'file_size', None),
            mime_type=getattr(job, 'mime_type', None),
            render_type=getattr(job, 'render_type', None),
            progress=getattr(job, 'progress', None),
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            error_message=getattr(job, 'error_message', None)
        )

        # Add extraction results if completed
        if job.status == JobStatus.COMPLETED and job.extraction_results:
            from ..models import ExtractionResult, ExtractedRow, Provenance
            
            rows = []
            for result in job.extraction_results:
                provenance = Provenance(
                    file=result.source_file,
                    page=result.source_page,
                    bbox=[result.source_bbox_x1, result.source_bbox_y1, 
                          result.source_bbox_x2, result.source_bbox_y2],
                    token_ids=result.token_ids,
                    confidence=result.confidence
                )
                
                row = ExtractedRow(
                    row_id=result.row_id,
                    page=result.page,
                    region_id=result.region_id,
                    bbox=[result.bbox_x1, result.bbox_y1, result.bbox_x2, result.bbox_y2],
                    columns=result.columns_data,
                    provenance=provenance,
                    needs_review=result.needs_review
                )
                rows.append(row)
            
            extraction_result = ExtractionResult(
                rows=rows,
                metadata=job.processing_metadata or {},
                processing_time=(job.completed_at - job.created_at).total_seconds() if job.completed_at else 0,
                render_type=job.render_type,
                confidence_score=job.confidence_score or 0.0
            )
            
            job_response.result = extraction_result
        
        return job_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job details for {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job details: {str(e)}"
        )


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Cancel a processing job.
    
    Only pending or processing jobs can be cancelled.
    """
    try:
        # Find the job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        # Check if job can be cancelled
        if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job with status: {job.status}"
            )
        
        # Update job status
        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.utcnow()
        db.commit()
        
        # TODO: Cancel the Celery task if it's running
        
        logger.info(f"Job {job_id} cancelled successfully")
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )


def _get_status_message(job: Job) -> str:
    """Get a human-readable status message for a job."""
    status_val = job.status
    try:
        status_enum = JobStatus(status_val)
    except Exception:
        # If stored as plain string, coerce where possible
        try:
            status_enum = JobStatus(status_val.value)  # unlikely branch
        except Exception:
            status_enum = JobStatus.PENDING

    progress = getattr(job, 'progress', 0.0) or 0.0

    if status_enum == JobStatus.PENDING:
        return "Job is queued for processing"
    elif status_enum == JobStatus.PROCESSING:
        if progress < 0.1:
            return "Analyzing document structure..."
        elif progress < 0.3:
            return "Performing OCR extraction..."
        elif progress < 0.7:
            return "Detecting and extracting tables..."
        elif progress < 0.9:
            return "Post-processing and validation..."
        else:
            return "Finalizing results..."
    elif status_enum == JobStatus.COMPLETED:
        return "Processing completed successfully"
    elif status_enum == JobStatus.FAILED:
        return getattr(job, 'error_message', None) or "Processing failed"
    else:
        return "Unknown status"
