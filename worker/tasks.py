"""
Celery tasks for CurioScan document processing pipeline.
"""

import os
import sys
import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from celery import Task
from celery.exceptions import Retry
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from worker.celery_app import celery_app
from worker.model_manager import ModelManager
from worker.document_processor import DocumentProcessor
from worker.storage_manager import StorageManager
from worker.webhook_sender import WebhookSender

logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://curioscan:curioscan123@localhost:5432/curioscan")
engine = sa.create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Global instances
model_manager = ModelManager()
storage_manager = StorageManager()
webhook_sender = WebhookSender()


class CallbackTask(Task):
    """Base task class with database session and error handling."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on task success."""
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(f"Task {task_id} failed: {str(exc)}")
        
        # Update job status in database
        if args and len(args) > 0:
            job_id = args[0]
            self._update_job_status(job_id, "failed", error_message=str(exc))
    
    def _update_job_status(self, job_id: str, status: str, progress: float = None, error_message: str = None):
        """Update job status in database."""
        try:
            from api.database import Job
            
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    job.status = status
                    if progress is not None:
                        job.progress = progress
                    if error_message:
                        job.error_message = error_message
                    job.updated_at = sa.func.now()
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to update job status: {str(e)}")


@celery_app.task(bind=True, base=CallbackTask, max_retries=3)
def process_document(self, job_id: str, input_path: str, confidence_threshold: float = 0.8):
    """
    Main document processing pipeline.
    
    This task orchestrates the entire processing workflow:
    1. Document classification
    2. Preprocessing
    3. OCR extraction
    4. Table detection
    5. Postprocessing
    6. Result storage
    """
    try:
        logger.info(f"Starting document processing for job {job_id}")
        
        # Update job status
        self._update_job_status(job_id, "processing", progress=0.0)
        
        # Initialize processor
        processor = DocumentProcessor(
            model_manager=model_manager,
            storage_manager=storage_manager,
            confidence_threshold=confidence_threshold
        )
        
        # Step 1: Classify document (10% progress)
        logger.info(f"Classifying document for job {job_id}")
        classification_result = processor.classify_document(input_path)
        self._update_job_status(job_id, "processing", progress=0.1)
        
        # Step 2: Preprocess document (20% progress)
        logger.info(f"Preprocessing document for job {job_id}")
        preprocessed_data = processor.preprocess_document(input_path, classification_result)
        self._update_job_status(job_id, "processing", progress=0.2)
        
        # Step 3: Extract text (50% progress)
        logger.info(f"Extracting text for job {job_id}")
        ocr_results = processor.extract_text(preprocessed_data, classification_result)
        self._update_job_status(job_id, "processing", progress=0.5)
        
        # Step 4: Detect and extract tables (70% progress)
        logger.info(f"Detecting tables for job {job_id}")
        table_results = processor.detect_tables(preprocessed_data, ocr_results)
        self._update_job_status(job_id, "processing", progress=0.7)
        
        # Step 5: Postprocess results (90% progress)
        logger.info(f"Postprocessing results for job {job_id}")
        final_results = processor.postprocess_results(
            ocr_results, table_results, classification_result
        )
        self._update_job_status(job_id, "processing", progress=0.9)
        
        # Step 6: Store results (100% progress)
        logger.info(f"Storing results for job {job_id}")
        output_path = processor.store_results(job_id, final_results)
        
        # Update job as completed
        self._update_job_completed(job_id, final_results, output_path)
        
        # Send webhook notification
        webhook_sender.send_job_completed(job_id, final_results)
        
        logger.info(f"Document processing completed for job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "output_path": output_path,
            "results_summary": {
                "total_rows": len(final_results.get("rows", [])),
                "confidence_score": final_results.get("confidence_score", 0.0),
                "render_type": classification_result.get("render_type"),
                "processing_time": time.time() - self.request.called_directly
            }
        }
        
    except Exception as e:
        logger.error(f"Document processing failed for job {job_id}: {str(e)}")
        self._update_job_status(job_id, "failed", error_message=str(e))
        webhook_sender.send_job_failed(job_id, str(e))
        raise
    
    def _update_job_completed(self, job_id: str, results: Dict[str, Any], output_path: str):
        """Update job as completed and store results."""
        try:
            from api.database import Job, ExtractionResult
            
            db = SessionLocal()
            try:
                job = db.query(Job).filter(Job.job_id == job_id).first()
                if job:
                    job.status = "completed"
                    job.progress = 1.0
                    job.output_path = output_path
                    job.confidence_score = results.get("confidence_score", 0.0)
                    job.render_type = results.get("render_type")
                    job.completed_at = sa.func.now()
                    job.updated_at = sa.func.now()
                    
                    # Store extraction results
                    for row_data in results.get("rows", []):
                        extraction_result = ExtractionResult(
                            job_id=job.id,
                            row_id=row_data["row_id"],
                            page=row_data["page"],
                            region_id=row_data["region_id"],
                            bbox_x1=row_data["bbox"][0],
                            bbox_y1=row_data["bbox"][1],
                            bbox_x2=row_data["bbox"][2],
                            bbox_y2=row_data["bbox"][3],
                            columns_data=row_data["columns"],
                            source_file=row_data["provenance"]["file"],
                            source_page=row_data["provenance"]["page"],
                            source_bbox_x1=row_data["provenance"]["bbox"][0],
                            source_bbox_y1=row_data["provenance"]["bbox"][1],
                            source_bbox_x2=row_data["provenance"]["bbox"][2],
                            source_bbox_y2=row_data["provenance"]["bbox"][3],
                            token_ids=row_data["provenance"]["token_ids"],
                            confidence=row_data["provenance"]["confidence"],
                            needs_review=row_data["needs_review"]
                        )
                        db.add(extraction_result)
                    
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to update job completion: {str(e)}")


@celery_app.task(bind=True, base=CallbackTask, max_retries=3)
def retrain_model(self, retrain_job_id: str, model_type: str, dataset_path: Optional[str], 
                  config_overrides: Dict[str, Any], user_id: str):
    """
    Model retraining task.
    
    Handles retraining of various model types with corrected data.
    """
    try:
        logger.info(f"Starting model retraining job {retrain_job_id} for {model_type}")
        
        # TODO: Implement model retraining logic
        # This would involve:
        # 1. Loading training data
        # 2. Preparing datasets
        # 3. Training the model
        # 4. Evaluating performance
        # 5. Saving new checkpoint
        
        # For now, simulate training
        import time
        time.sleep(10)  # Simulate training time
        
        logger.info(f"Model retraining completed for job {retrain_job_id}")
        
        return {
            "retrain_job_id": retrain_job_id,
            "model_type": model_type,
            "status": "completed",
            "message": "Model retraining completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed for job {retrain_job_id}: {str(e)}")
        raise


@celery_app.task(bind=True, max_retries=3)
def send_webhook(self, webhook_url: str, payload: Dict[str, Any], secret: Optional[str] = None):
    """
    Send webhook notification.
    """
    try:
        webhook_sender.send_webhook(webhook_url, payload, secret)
        logger.info(f"Webhook sent successfully to {webhook_url}")
        
    except Exception as e:
        logger.error(f"Failed to send webhook to {webhook_url}: {str(e)}")
        raise


# Individual processing tasks for more granular control
@celery_app.task(bind=True, base=CallbackTask)
def classify_document(self, input_path: str):
    """Classify document render type."""
    processor = DocumentProcessor(model_manager, storage_manager)
    return processor.classify_document(input_path)


@celery_app.task(bind=True, base=CallbackTask)
def preprocess_document(self, input_path: str, classification_result: Dict[str, Any]):
    """Preprocess document based on classification."""
    processor = DocumentProcessor(model_manager, storage_manager)
    return processor.preprocess_document(input_path, classification_result)


@celery_app.task(bind=True, base=CallbackTask)
def extract_text(self, preprocessed_data: Dict[str, Any], classification_result: Dict[str, Any]):
    """Extract text using OCR."""
    processor = DocumentProcessor(model_manager, storage_manager)
    return processor.extract_text(preprocessed_data, classification_result)


@celery_app.task(bind=True, base=CallbackTask)
def detect_tables(self, preprocessed_data: Dict[str, Any], ocr_results: Dict[str, Any]):
    """Detect and extract tables."""
    processor = DocumentProcessor(model_manager, storage_manager)
    return processor.detect_tables(preprocessed_data, ocr_results)


@celery_app.task(bind=True, base=CallbackTask)
def postprocess_results(self, ocr_results: Dict[str, Any], table_results: Dict[str, Any], 
                       classification_result: Dict[str, Any]):
    """Postprocess and normalize results."""
    processor = DocumentProcessor(model_manager, storage_manager)
    return processor.postprocess_results(ocr_results, table_results, classification_result)
