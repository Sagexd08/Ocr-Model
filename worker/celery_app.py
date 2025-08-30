import os

from celery import Celery

# Use SQLite DB transport for broker and results to avoid Redis
BROKER_URL = os.getenv("CELERY_BROKER_URL", "sqla+sqlite:///celerydb.sqlite")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "db+sqlite:///celerydb.sqlite")

# Removed legacy Redis-based app init below
# (keep this file minimal for SQLite transport)


celery_app = Celery(
    "worker",
    broker=BROKER_URL,
    backend=BACKEND_URL
)