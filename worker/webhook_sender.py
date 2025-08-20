"""
Webhook sender for CurioScan workers.

Handles sending webhook notifications for job events.
"""

import os
import logging
import json
import hashlib
import hmac
from typing import Dict, Any, Optional, List
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class WebhookSender:
    """Handles webhook notifications."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CurioScan-Webhook/1.0",
            "Content-Type": "application/json"
        })
    
    def send_job_created(self, job_id: str, job_data: Dict[str, Any]):
        """Send job created webhook."""
        payload = {
            "event": "job.created",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": job_data
        }
        
        self._send_webhooks(payload)
    
    def send_job_started(self, job_id: str):
        """Send job started webhook."""
        payload = {
            "event": "job.started",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_webhooks(payload)
    
    def send_job_progress(self, job_id: str, progress: float, message: str = None):
        """Send job progress webhook."""
        payload = {
            "event": "job.progress",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "progress": progress,
                "message": message
            }
        }
        
        self._send_webhooks(payload)
    
    def send_job_completed(self, job_id: str, results: Dict[str, Any]):
        """Send job completed webhook."""
        payload = {
            "event": "job.completed",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "total_rows": len(results.get("rows", [])),
                "confidence_score": results.get("confidence_score", 0.0),
                "render_type": results.get("render_type"),
                "rows_needing_review": sum(1 for row in results.get("rows", []) 
                                         if row.get("needs_review", False))
            }
        }
        
        self._send_webhooks(payload)
    
    def send_job_failed(self, job_id: str, error_message: str):
        """Send job failed webhook."""
        payload = {
            "event": "job.failed",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "error": error_message
            }
        }
        
        self._send_webhooks(payload)
    
    def send_job_cancelled(self, job_id: str):
        """Send job cancelled webhook."""
        payload = {
            "event": "job.cancelled",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_webhooks(payload)
    
    def send_webhook(self, url: str, payload: Dict[str, Any], secret: Optional[str] = None):
        """Send a single webhook."""
        try:
            headers = {}
            
            # Add signature if secret is provided
            if secret:
                signature = self._generate_signature(payload, secret)
                headers["X-CurioScan-Signature"] = signature
            
            # Add timestamp
            headers["X-CurioScan-Timestamp"] = str(int(datetime.utcnow().timestamp()))
            
            response = self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            logger.info(f"Webhook sent successfully to {url}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send webhook to {url}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending webhook to {url}: {str(e)}")
            return False
    
    def _send_webhooks(self, payload: Dict[str, Any]):
        """Send webhook to all registered endpoints."""
        try:
            # Get registered webhooks from database
            webhooks = self._get_registered_webhooks(payload["event"])
            
            for webhook in webhooks:
                try:
                    success = self.send_webhook(
                        webhook["url"],
                        payload,
                        webhook.get("secret")
                    )
                    
                    # Update webhook statistics
                    self._update_webhook_stats(webhook["webhook_id"], success)
                    
                except Exception as e:
                    logger.error(f"Failed to send webhook {webhook['webhook_id']}: {str(e)}")
                    self._update_webhook_stats(webhook["webhook_id"], False)
        
        except Exception as e:
            logger.error(f"Failed to send webhooks for event {payload['event']}: {str(e)}")
    
    def _get_registered_webhooks(self, event: str) -> List[Dict[str, Any]]:
        """Get registered webhooks for an event."""
        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            sys.path.append(str(project_root))
            
            from api.database import SessionLocal, Webhook
            
            db = SessionLocal()
            try:
                webhooks = db.query(Webhook).filter(
                    Webhook.active == True,
                    Webhook.events.contains([event])
                ).all()
                
                return [
                    {
                        "webhook_id": webhook.webhook_id,
                        "url": webhook.url,
                        "secret": webhook.secret
                    }
                    for webhook in webhooks
                ]
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to get registered webhooks: {str(e)}")
            return []
    
    def _update_webhook_stats(self, webhook_id: str, success: bool):
        """Update webhook call statistics."""
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            sys.path.append(str(project_root))
            
            from api.database import SessionLocal, Webhook
            import sqlalchemy as sa
            
            db = SessionLocal()
            try:
                webhook = db.query(Webhook).filter(
                    Webhook.webhook_id == webhook_id
                ).first()
                
                if webhook:
                    webhook.total_calls += 1
                    if success:
                        webhook.successful_calls += 1
                    else:
                        webhook.failed_calls += 1
                    webhook.last_called_at = sa.func.now()
                    
                    db.commit()
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to update webhook stats: {str(e)}")
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
