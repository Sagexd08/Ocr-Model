"""
Celery tasks for document processing.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Union

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from celery.utils.log import get_task_logger

from ..api.celery_app import celery_app
from .model_manager import ModelManager
from .storage_manager import StorageManager
from .document_processor import EnhancedDocumentProcessor
from .types import Document, Page, JobStatus, ProcessingMode
from ..api.database import SessionLocal
from ..api import models

# Setup logging
logger = get_task_logger(__name__)

# Initialize managers and processors
model_manager = ModelManager()
storage_manager = StorageManager()
document_processor = EnhancedDocumentProcessor(model_manager, storage_manager)

@celery_app.task(bind=True)
def process_document_legacy(self, file_name):
    """Legacy document processing task"""
    storage_manager = StorageManager()
    file_data = storage_manager.get_file(file_name)
    
    mime_type = f"application/{file_name.split('.')[-1]}"

    doc = Document(file_name=file_name, mime_type=mime_type, pages=[])
    processor = DocumentProcessor(doc, file_data.read())
    result = processor.process()

    db = SessionLocal()
    db_document = models.Document(job_id=self.request.id, content=result)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    db.close()

    return result

@celery_app.task(
    name="process_document",
    bind=True,
    max_retries=3,
    soft_time_limit=600,  # 10 minutes
    time_limit=900,       # 15 minutes
    retry_backoff=True,
    retry_backoff_max=300,  # 5 minutes max delay
    retry_jitter=True,
)
def process_document(self, job_id: str, document_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document asynchronously.
    
    Args:
        job_id: Unique job identifier
        document_path: Path to the document
        params: Processing parameters
        
    Returns:
        Processing results
    """
    logger.info(f"Starting document processing job {job_id}")
    
    try:
        # Update job status to processing
        _update_job_status(job_id, JobStatus.PROCESSING)
        
        # Process document
        result = document_processor.process_document(job_id, document_path, params)
        
        # Update job status based on result
        if result["status"] == JobStatus.COMPLETED:
            _update_job_status(job_id, JobStatus.COMPLETED, result.get("result_path"))
        else:
            _update_job_status(job_id, JobStatus.FAILED, error=result.get("error"))
        
        # Save to database
        db = SessionLocal()
        db_document = models.Document(job_id=job_id, content=result)
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        db.close()
        
        return result
        
    except SoftTimeLimitExceeded:
        logger.error(f"Job {job_id} exceeded time limit")
        _update_job_status(job_id, JobStatus.FAILED, error="Processing time limit exceeded")
        raise
        
    except Exception as e:
        logger.exception(f"Error processing document for job {job_id}: {str(e)}")
        _update_job_status(job_id, JobStatus.FAILED, error=str(e))
        
        # Retry with exponential backoff
        self.retry(exc=e)

@celery_app.task(name="cleanup_temp_files")
def cleanup_temp_files() -> Dict[str, Any]:
    """
    Cleanup temporary files older than the specified retention period.
    
    Returns:
        Cleanup results
    """
    logger.info("Starting cleanup of temporary files")
    
    try:
        # Get temporary directory
        temp_dir = storage_manager.get_temp_dir()
        
        # Get retention period in days (default to 1 day)
        retention_days = int(os.environ.get("TEMP_FILES_RETENTION_DAYS", "1"))
        retention_seconds = retention_days * 24 * 60 * 60
        
        # Get current time
        current_time = time.time()
        
        # Track statistics
        stats = {
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": 0
        }
        
        # Walk through temp directory
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                try:
                    # Get file stats
                    file_stat = os.stat(file_path)
                    
                    # Check if file is older than retention period
                    if current_time - file_stat.st_mtime > retention_seconds:
                        # Delete file
                        stats["bytes_freed"] += file_stat.st_size
                        os.unlink(file_path)
                        stats["files_deleted"] += 1
                        
                except Exception as e:
                    logger.error(f"Error cleaning up file {file_path}: {str(e)}")
                    stats["errors"] += 1
        
        logger.info(f"Cleanup completed: {stats['files_deleted']} files deleted, {stats['bytes_freed']/1024/1024:.2f} MB freed")
        return stats
        
    except Exception as e:
        logger.exception(f"Error during cleanup: {str(e)}")
        return {
            "error": str(e),
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": 1
        }

@celery_app.task(name="regenerate_export", bind=True, max_retries=2)
def regenerate_export(self, job_id: str, output_format: str) -> Dict[str, Any]:
    """
    Regenerate export in the specified format from existing processing results.
    
    Args:
        job_id: Unique job identifier
        output_format: Output format (json, csv, excel, text)
        
    Returns:
        Export results
    """
    logger.info(f"Regenerating {output_format} export for job {job_id}")
    
    try:
        # Get the existing results
        results_path = storage_manager.get_result_path(job_id) + ".json"
        
        if not os.path.exists(results_path):
            error = f"Results not found for job {job_id}"
            logger.error(error)
            return {"error": error}
        
        # Load the results
        import json
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Generate the new export
        export_path = document_processor._save_results(job_id, results, output_format)
        
        return {
            "job_id": job_id,
            "status": "success",
            "export_path": export_path,
            "format": output_format
        }
        
    except Exception as e:
        logger.exception(f"Error regenerating export for job {job_id}: {str(e)}")
        self.retry(exc=e)

def _update_job_status(job_id: str, status: JobStatus, result_path: Optional[str] = None, error: Optional[str] = None) -> None:
    """
    Update job status in the database.
    
    Args:
        job_id: Job identifier
        status: New job status
        result_path: Path to the results (for completed jobs)
        error: Error message (for failed jobs)
    """
    # Update status in database
    db = SessionLocal()
    try:
        job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
        if job:
            job.status = status.value
            # Persist result path in processing_metadata for now (no dedicated column)
            if result_path:
                meta = job.processing_metadata or {}
                if isinstance(meta, dict):
                    meta["result_path"] = result_path
                    job.processing_metadata = meta
            if error:
                job.error_message = error
            db.commit()
    finally:
        db.close()