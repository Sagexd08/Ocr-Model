"""
Celery application configuration for CurioScan API.
"""

from celery import Celery
from .config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "curioscan",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
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
    worker_max_tasks_per_child=1000,
)

# Task routing
celery_app.conf.task_routes = {
    "worker.tasks.process_document": {"queue": "default"},
    "worker.tasks.retrain_model": {"queue": "training"},
    "worker.tasks.send_webhook": {"queue": "webhooks"},
}
