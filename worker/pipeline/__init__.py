from typing import Dict, Any, Optional
import uuid
import logging

from ..types import Document
from ..utils.logging import get_logger

logger = get_logger(__name__)

def _get_confidence_threshold(config: Dict[str, Any]) -> float:
    """Get confidence threshold from config with defaults"""
    return config.get("confidence_threshold", 0.7)

class DocumentProcessingError(Exception):
    """Exception raised for document processing errors with context"""
    
    def __init__(
        self, 
        message: str, 
        job_id: str = None, 
        stage: str = None,
        details: str = None
    ):
        self.message = message
        self.job_id = job_id
        self.stage = stage
        self.details = details
        super().__init__(self.message)
