"""
Common feature flags for API and Worker.
Values are derived from environment variables so both processes can share behavior.
"""
import os

# Defaults are permissive (enabled) for production; disable as needed in dev/test
LOAD_MODELS_ON_STARTUP: bool = os.getenv("LOAD_MODELS_ON_STARTUP", "1") == "1"
LOAD_EASYOCR: bool = os.getenv("LOAD_EASYOCR", "1") == "1"
LOAD_TABLE_DETECTOR: bool = os.getenv("LOAD_TABLE_DETECTOR", "1") == "1"
LOAD_LAYOUT_ANALYZER: bool = os.getenv("LOAD_LAYOUT_ANALYZER", "1") == "1"

__all__ = [
    "LOAD_MODELS_ON_STARTUP",
    "LOAD_EASYOCR",
    "LOAD_TABLE_DETECTOR",
    "LOAD_LAYOUT_ANALYZER",
]

