"""
Model retraining endpoints for CurioScan API.
"""

import uuid
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db, ModelCheckpoint
from ..models import RetrainRequest, RetrainResponse
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/retrain-trigger", response_model=RetrainResponse)
async def trigger_retraining(
    retrain_request: RetrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Trigger model retraining with corrected data.
    
    Supports retraining of:
    - renderer_classifier: Document type classification
    - ocr_models: OCR accuracy improvement
    - table_detector: Table detection and extraction
    - layout_analyzer: Document layout understanding
    """
    try:
        # Validate model type
        valid_model_types = [
            "renderer_classifier",
            "ocr_models", 
            "table_detector",
            "layout_analyzer"
        ]
        
        if retrain_request.model_type not in valid_model_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {retrain_request.model_type}. "
                       f"Valid types: {valid_model_types}"
            )
        
        # Generate retraining job ID
        retrain_job_id = str(uuid.uuid4())
        
        logger.info(f"Starting retraining job {retrain_job_id} for model {retrain_request.model_type}")
        
        # Estimate duration based on model type
        duration_estimates = {
            "renderer_classifier": 30,  # 30 minutes
            "ocr_models": 120,          # 2 hours
            "table_detector": 180,      # 3 hours
            "layout_analyzer": 240      # 4 hours
        }
        
        estimated_duration = duration_estimates.get(retrain_request.model_type, 60)
        
        if retrain_request.dry_run:
            # Dry run mode - just validate and return
            return RetrainResponse(
                retrain_job_id=retrain_job_id,
                status="dry_run_completed",
                estimated_duration=estimated_duration,
                message=f"Dry run completed for {retrain_request.model_type}. "
                       f"Estimated training time: {estimated_duration} minutes"
            )
        
        # Queue the retraining task
        background_tasks.add_task(
            queue_retraining_task,
            retrain_job_id,
            retrain_request.model_type,
            retrain_request.dataset_path,
            retrain_request.config_overrides,
            current_user.get("user_id") if current_user else "anonymous"
        )
        
        return RetrainResponse(
            retrain_job_id=retrain_job_id,
            status="queued",
            estimated_duration=estimated_duration,
            message=f"Retraining job queued for {retrain_request.model_type}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger retraining: {str(e)}"
        )


async def queue_retraining_task(
    retrain_job_id: str,
    model_type: str,
    dataset_path: Optional[str],
    config_overrides: Dict[str, Any],
    user_id: str
):
    """Queue the model retraining task."""
    try:
        # Import here to avoid circular imports
        from ..celery_app import celery_app
        
        # Queue the retraining task
        task = celery_app.send_task(
            'worker.tasks.retrain_model',
            args=[retrain_job_id, model_type, dataset_path, config_overrides, user_id],
            queue='training'  # Use dedicated training queue
        )
        
        logger.info(f"Queued retraining task {task.id} for job {retrain_job_id}")
        
    except Exception as e:
        logger.error(f"Failed to queue retraining task for job {retrain_job_id}: {str(e)}")


@router.get("/retrain/status/{retrain_job_id}")
async def get_retraining_status(
    retrain_job_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Get the status of a retraining job."""
    try:
        # TODO: Implement retraining job status tracking
        # This would involve checking Celery task status and training progress
        
        return {
            "retrain_job_id": retrain_job_id,
            "status": "not_implemented",
            "message": "Retraining status tracking not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Failed to get retraining status for {retrain_job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get retraining status: {str(e)}"
        )


@router.get("/models/checkpoints")
async def list_model_checkpoints(
    model_name: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """List available model checkpoints."""
    try:
        query = db.query(ModelCheckpoint)
        
        if model_name:
            query = query.filter(ModelCheckpoint.model_name == model_name)
        
        if is_active is not None:
            query = query.filter(ModelCheckpoint.is_active == is_active)
        
        checkpoints = query.order_by(ModelCheckpoint.created_at.desc()).all()
        
        checkpoint_list = []
        for checkpoint in checkpoints:
            checkpoint_info = {
                "id": str(checkpoint.id),
                "model_name": checkpoint.model_name,
                "version": checkpoint.version,
                "checkpoint_path": checkpoint.checkpoint_path,
                "metrics": checkpoint.metrics,
                "is_active": checkpoint.is_active,
                "is_production": checkpoint.is_production,
                "created_at": checkpoint.created_at,
                "training_duration": checkpoint.training_duration
            }
            checkpoint_list.append(checkpoint_info)
        
        return {
            "checkpoints": checkpoint_list,
            "total_count": len(checkpoint_list)
        }
        
    except Exception as e:
        logger.error(f"Failed to list model checkpoints: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list model checkpoints: {str(e)}"
        )


@router.post("/models/activate/{checkpoint_id}")
async def activate_model_checkpoint(
    checkpoint_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Activate a specific model checkpoint."""
    try:
        # Find the checkpoint
        checkpoint = db.query(ModelCheckpoint).filter(
            ModelCheckpoint.id == checkpoint_id
        ).first()
        
        if not checkpoint:
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint {checkpoint_id} not found"
            )
        
        # Deactivate other checkpoints for the same model
        db.query(ModelCheckpoint).filter(
            ModelCheckpoint.model_name == checkpoint.model_name,
            ModelCheckpoint.is_active == True
        ).update({"is_active": False})
        
        # Activate the selected checkpoint
        checkpoint.is_active = True
        
        db.commit()
        
        logger.info(f"Activated checkpoint {checkpoint_id} for model {checkpoint.model_name}")
        
        return {
            "message": f"Checkpoint {checkpoint_id} activated successfully",
            "model_name": checkpoint.model_name,
            "version": checkpoint.version
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate checkpoint {checkpoint_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to activate checkpoint: {str(e)}"
        )


@router.get("/retrain/datasets")
async def list_training_datasets(
    current_user: Optional[dict] = Depends(get_current_user)
):
    """List available training datasets for retraining."""
    try:
        # TODO: Implement dataset discovery
        # This would scan for available training datasets and return metadata
        
        datasets = [
            {
                "name": "corrected_extractions",
                "description": "Dataset built from human-corrected OCR extractions",
                "size": "unknown",
                "last_updated": "unknown"
            },
            {
                "name": "synthetic_documents", 
                "description": "Synthetically generated training documents",
                "size": "unknown",
                "last_updated": "unknown"
            }
        ]
        
        return {
            "datasets": datasets,
            "total_count": len(datasets)
        }
        
    except Exception as e:
        logger.error(f"Failed to list training datasets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list training datasets: {str(e)}"
        )


@router.post("/retrain/export-corrections")
async def export_corrections_dataset(
    job_ids: Optional[list] = None,
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Export human corrections as a training dataset.
    
    Creates a dataset from reviewed and corrected OCR extractions.
    """
    try:
        # TODO: Implement corrections export
        # This would:
        # 1. Query reviewed items with corrections
        # 2. Format them as training data
        # 3. Save to a dataset file
        # 4. Return download link
        
        return {
            "message": "Corrections export not yet implemented",
            "status": "not_implemented"
        }
        
    except Exception as e:
        logger.error(f"Failed to export corrections dataset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export corrections dataset: {str(e)}"
        )
