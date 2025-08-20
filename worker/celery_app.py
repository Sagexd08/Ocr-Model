"""
Celery application for CurioScan workers.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from celery import Celery
from celery.signals import worker_ready, worker_shutdown
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://curioscan:curioscan123@localhost:5432/curioscan")

# Create Celery app
celery_app = Celery(
    "curioscan_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_reject_on_worker_lost=True,
)

# Task routing
celery_app.conf.task_routes = {
    "worker.tasks.process_document": {"queue": "default"},
    "worker.tasks.retrain_model": {"queue": "training"},
    "worker.tasks.send_webhook": {"queue": "webhooks"},
    "worker.tasks.classify_document": {"queue": "default"},
    "worker.tasks.preprocess_document": {"queue": "default"},
    "worker.tasks.extract_text": {"queue": "default"},
    "worker.tasks.detect_tables": {"queue": "default"},
    "worker.tasks.postprocess_results": {"queue": "default"},
}


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handler called when worker is ready."""
    logger.info("CurioScan worker is ready")
    
    # Initialize models and dependencies
    try:
        from worker.model_manager import ModelManager
        model_manager = ModelManager()
        model_manager.initialize_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handler called when worker is shutting down."""
    logger.info("CurioScan worker is shutting down")


if __name__ == "__main__":
    celery_app.start()
