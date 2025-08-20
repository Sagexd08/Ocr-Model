"""
Human review endpoints for CurioScan API.
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..config import get_settings
from ..database import get_db, Job, ReviewItem as DBReviewItem
from ..models import ReviewResponse, ReviewItem, ReviewUpdate, JobStatus
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/review/{job_id}", response_model=ReviewResponse)
async def get_review_items(
    job_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by review status"),
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Get items requiring human review for a specific job.
    
    Returns paginated list of items with low confidence or requiring validation.
    """
    try:
        # Find the job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        # Build query for review items
        query = db.query(DBReviewItem).filter(DBReviewItem.job_id == job.id)
        
        # Filter by status if provided
        if status:
            query = query.filter(DBReviewItem.status == status)
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        review_items_db = query.order_by(desc(DBReviewItem.created_at)).offset(offset).limit(page_size).all()
        
        # Convert to response models
        review_items = []
        for item in review_items_db:
            review_item = ReviewItem(
                item_id=item.item_id,
                job_id=job_id,
                page=item.page,
                bbox=[item.bbox_x1, item.bbox_y1, item.bbox_x2, item.bbox_y2],
                original_text=item.original_text,
                confidence=item.confidence,
                suggested_text=item.suggested_text,
                context=item.context_data or {}
            )
            review_items.append(review_item)
        
        return ReviewResponse(
            items=review_items,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get review items for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get review items: {str(e)}"
        )


@router.post("/review/update")
async def update_review_items(
    updates: List[ReviewUpdate],
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Update review items with human corrections.
    
    Accepts a list of review updates with actions: accept, reject, edit.
    """
    try:
        updated_items = []
        
        for update in updates:
            # Find the review item
            review_item = db.query(DBReviewItem).filter(
                DBReviewItem.item_id == update.item_id
            ).first()
            
            if not review_item:
                logger.warning(f"Review item {update.item_id} not found")
                continue
            
            # Update based on action
            if update.action == "accept":
                review_item.status = "accepted"
                review_item.corrected_text = review_item.original_text
            elif update.action == "reject":
                review_item.status = "rejected"
                review_item.corrected_text = None
            elif update.action == "edit":
                if not update.corrected_text:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Corrected text required for edit action on item {update.item_id}"
                    )
                review_item.status = "edited"
                review_item.corrected_text = update.corrected_text
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid action: {update.action}. Valid actions: accept, reject, edit"
                )
            
            # Update metadata
            review_item.reviewed_at = datetime.utcnow()
            review_item.reviewed_by = current_user.get("user_id") if current_user else "anonymous"
            review_item.updated_at = datetime.utcnow()
            
            updated_items.append(review_item.item_id)
        
        db.commit()
        
        logger.info(f"Updated {len(updated_items)} review items")
        
        return {
            "message": f"Successfully updated {len(updated_items)} review items",
            "updated_items": updated_items
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update review items: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update review items: {str(e)}"
        )


@router.get("/review/stats/{job_id}")
async def get_review_stats(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """Get review statistics for a job."""
    try:
        # Find the job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        # Get review statistics
        total_items = db.query(DBReviewItem).filter(DBReviewItem.job_id == job.id).count()
        pending_items = db.query(DBReviewItem).filter(
            DBReviewItem.job_id == job.id,
            DBReviewItem.status == "pending"
        ).count()
        accepted_items = db.query(DBReviewItem).filter(
            DBReviewItem.job_id == job.id,
            DBReviewItem.status == "accepted"
        ).count()
        rejected_items = db.query(DBReviewItem).filter(
            DBReviewItem.job_id == job.id,
            DBReviewItem.status == "rejected"
        ).count()
        edited_items = db.query(DBReviewItem).filter(
            DBReviewItem.job_id == job.id,
            DBReviewItem.status == "edited"
        ).count()
        
        return {
            "job_id": job_id,
            "total_items": total_items,
            "pending_items": pending_items,
            "accepted_items": accepted_items,
            "rejected_items": rejected_items,
            "edited_items": edited_items,
            "completion_rate": (accepted_items + rejected_items + edited_items) / total_items if total_items > 0 else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get review stats for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get review stats: {str(e)}"
        )


@router.get("/review/queue")
async def get_review_queue(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Get global review queue across all jobs.
    
    Returns items that need review, prioritized by confidence score and age.
    """
    try:
        # Build query for pending review items
        query = db.query(DBReviewItem).filter(DBReviewItem.status == "pending")
        
        # Order by confidence (lowest first) and creation time (oldest first)
        query = query.order_by(DBReviewItem.confidence.asc(), DBReviewItem.created_at.asc())
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        review_items_db = query.offset(offset).limit(page_size).all()
        
        # Convert to response models
        review_items = []
        for item in review_items_db:
            # Get job info
            job = db.query(Job).filter(Job.id == item.job_id).first()
            
            review_item = ReviewItem(
                item_id=item.item_id,
                job_id=job.job_id if job else "unknown",
                page=item.page,
                bbox=[item.bbox_x1, item.bbox_y1, item.bbox_x2, item.bbox_y2],
                original_text=item.original_text,
                confidence=item.confidence,
                suggested_text=item.suggested_text,
                context={
                    **(item.context_data or {}),
                    "file_name": job.file_name if job else "unknown",
                    "created_at": item.created_at.isoformat()
                }
            )
            review_items.append(review_item)
        
        return ReviewResponse(
            items=review_items,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to get review queue: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get review queue: {str(e)}"
        )


@router.post("/review/bulk-accept")
async def bulk_accept_reviews(
    job_id: str,
    confidence_threshold: float = Query(0.9, ge=0.0, le=1.0, description="Accept items above this confidence"),
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Bulk accept review items above a confidence threshold.
    
    Useful for accepting high-confidence items in batch.
    """
    try:
        # Find the job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        # Find items to accept
        items_to_accept = db.query(DBReviewItem).filter(
            DBReviewItem.job_id == job.id,
            DBReviewItem.status == "pending",
            DBReviewItem.confidence >= confidence_threshold
        ).all()
        
        # Update items
        updated_count = 0
        for item in items_to_accept:
            item.status = "accepted"
            item.corrected_text = item.original_text
            item.reviewed_at = datetime.utcnow()
            item.reviewed_by = current_user.get("user_id") if current_user else "anonymous"
            item.updated_at = datetime.utcnow()
            updated_count += 1
        
        db.commit()
        
        logger.info(f"Bulk accepted {updated_count} items for job {job_id}")
        
        return {
            "message": f"Successfully accepted {updated_count} items",
            "accepted_count": updated_count,
            "confidence_threshold": confidence_threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to bulk accept reviews for job {job_id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to bulk accept reviews: {str(e)}"
        )
