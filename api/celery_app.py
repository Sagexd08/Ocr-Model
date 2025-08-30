"""
Celery application configuration for CurioScan API.
"""

from celery import Celery
from .config import get_settings

settings = get_settings()

# Broker/backend resolved from env with safe SQLite defaults
import os
broker_url = os.getenv("CELERY_BROKER_URL", "sqla+sqlite:///celerydb.sqlite")
backend_url = os.getenv("CELERY_RESULT_BACKEND", "db+sqlite:///celerydb.sqlite")

# Create Celery app
celery_app = Celery(
    "curioscan",
    broker=broker_url,
    backend=backend_url,
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

# Eager mode support: only apply auto-fallback for Redis brokers
import os
force_eager = os.getenv("CELERY_EAGER", "0") == "1"
if force_eager:
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

# Task routing
celery_app.conf.task_routes = {
    "worker.tasks.process_document": {"queue": "default"},
    "worker.tasks.retrain_model": {"queue": "training"},
    "worker.tasks.send_webhook": {"queue": "webhooks"},
}
