"""
Webhook management endpoints for CurioScan API.
"""

import uuid
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db, Webhook
from ..models import WebhookRequest, WebhookResponse
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/webhooks/register", response_model=WebhookResponse)
async def register_webhook(
    webhook_request: WebhookRequest,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Register a webhook for job status notifications.
    
    Supported events:
    - job.created
    - job.started
    - job.progress
    - job.completed
    - job.failed
    - job.cancelled
    """
    try:
        # Validate webhook URL
        if not webhook_request.url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail="Webhook URL must start with http:// or https://"
            )
        
        # Validate events
        valid_events = {
            "job.created", "job.started", "job.progress", 
            "job.completed", "job.failed", "job.cancelled"
        }
        
        invalid_events = set(webhook_request.events) - valid_events
        if invalid_events:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid events: {invalid_events}. Valid events: {valid_events}"
            )
        
        # Generate webhook ID
        webhook_id = str(uuid.uuid4())
        
        # Create webhook record
        webhook = Webhook(
            webhook_id=webhook_id,
            url=webhook_request.url,
            events=webhook_request.events,
            secret=webhook_request.secret,
            active=True
        )
        
        db.add(webhook)
        db.commit()
        db.refresh(webhook)
        
        logger.info(f"Webhook {webhook_id} registered for URL: {webhook_request.url}")
        
        return WebhookResponse(
            webhook_id=webhook_id,
            url=webhook_request.url,
            events=webhook_request.events,
            created_at=webhook.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register webhook: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register webhook: {str(e)}"
        )


@router.get("/webhooks", response_model=List[WebhookResponse])
async def list_webhooks(
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """List all registered webhooks."""
    try:
        webhooks = db.query(Webhook).filter(Webhook.active == True).all()
        
        webhook_responses = []
        for webhook in webhooks:
            webhook_response = WebhookResponse(
                webhook_id=webhook.webhook_id,
                url=webhook.url,
                events=webhook.events,
                created_at=webhook.created_at
            )
            webhook_responses.append(webhook_response)
        
        return webhook_responses
        
    except Exception as e:
        logger.error(f"Failed to list webhooks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list webhooks: {str(e)}"
        )


@router.get("/webhooks/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Get details of a specific webhook."""
    try:
        webhook = db.query(Webhook).filter(
            Webhook.webhook_id == webhook_id,
            Webhook.active == True
        ).first()
        
        if not webhook:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook {webhook_id} not found"
            )
        
        return WebhookResponse(
            webhook_id=webhook.webhook_id,
            url=webhook.url,
            events=webhook.events,
            created_at=webhook.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get webhook {webhook_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get webhook: {str(e)}"
        )


@router.put("/webhooks/{webhook_id}")
async def update_webhook(
    webhook_id: str,
    webhook_request: WebhookRequest,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Update an existing webhook."""
    try:
        webhook = db.query(Webhook).filter(
            Webhook.webhook_id == webhook_id,
            Webhook.active == True
        ).first()
        
        if not webhook:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook {webhook_id} not found"
            )
        
        # Validate webhook URL
        if not webhook_request.url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail="Webhook URL must start with http:// or https://"
            )
        
        # Validate events
        valid_events = {
            "job.created", "job.started", "job.progress", 
            "job.completed", "job.failed", "job.cancelled"
        }
        
        invalid_events = set(webhook_request.events) - valid_events
        if invalid_events:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid events: {invalid_events}. Valid events: {valid_events}"
            )
        
        # Update webhook
        webhook.url = webhook_request.url
        webhook.events = webhook_request.events
        webhook.secret = webhook_request.secret
        webhook.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Webhook {webhook_id} updated")
        
        return {"message": f"Webhook {webhook_id} updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update webhook {webhook_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update webhook: {str(e)}"
        )


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Delete (deactivate) a webhook."""
    try:
        webhook = db.query(Webhook).filter(
            Webhook.webhook_id == webhook_id,
            Webhook.active == True
        ).first()
        
        if not webhook:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook {webhook_id} not found"
            )
        
        # Deactivate webhook (soft delete)
        webhook.active = False
        webhook.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Webhook {webhook_id} deleted")
        
        return {"message": f"Webhook {webhook_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete webhook {webhook_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete webhook: {str(e)}"
        )


@router.get("/webhooks/{webhook_id}/stats")
async def get_webhook_stats(
    webhook_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Get webhook call statistics."""
    try:
        webhook = db.query(Webhook).filter(
            Webhook.webhook_id == webhook_id,
            Webhook.active == True
        ).first()
        
        if not webhook:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook {webhook_id} not found"
            )
        
        return {
            "webhook_id": webhook.webhook_id,
            "total_calls": webhook.total_calls,
            "successful_calls": webhook.successful_calls,
            "failed_calls": webhook.failed_calls,
            "success_rate": webhook.successful_calls / webhook.total_calls if webhook.total_calls > 0 else 0,
            "last_called_at": webhook.last_called_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get webhook stats for {webhook_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get webhook stats: {str(e)}"
        )
