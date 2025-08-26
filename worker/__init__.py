"""
CurioScan Worker Package

Celery-based worker pipeline for document processing.
"""

__version__ = "1.0.0"

# Re-export tasks for tests that import worker.tasks
try:
    from . import tasks
except Exception:
    tasks = None

