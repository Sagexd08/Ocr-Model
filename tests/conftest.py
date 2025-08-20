"""
Pytest configuration and fixtures for CurioScan tests.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import json

import pytest
import torch
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        "model": {
            "type": "renderer_classifier",
            "num_classes": 8,
            "backbone": "resnet18"
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "device": "cpu"
        },
        "data": {
            "batch_size": 4,
            "num_workers": 0,
            "target_size": [224, 224]
        },
        "evaluation": {
            "confidence_thresholds": [0.5, 0.8, 0.9],
            "save_visualizations": False
        }
    }


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image() -> Image.Image:
    """Sample test image."""
    # Create a simple test image
    image = Image.new('RGB', (224, 224), color='white')
    
    # Add some simple content
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 174, 174], fill='blue')
    draw.text((60, 60), "Test Document", fill='black')
    
    return image


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Sample PDF file as bytes."""
    # Create a minimal PDF
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.drawString(100, 750, "Test PDF Document")
        p.drawString(100, 700, "This is a test document for CurioScan")
        p.showPage()
        p.save()
        
        buffer.seek(0)
        return buffer.read()
        
    except ImportError:
        # Fallback: create a fake PDF header
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"


@pytest.fixture
def sample_ocr_results() -> Dict[str, Any]:
    """Sample OCR results."""
    return {
        "tokens": [
            {
                "text": "Test",
                "bbox": [10, 10, 50, 30],
                "confidence": 0.95,
                "token_id": 1
            },
            {
                "text": "Document",
                "bbox": [60, 10, 120, 30],
                "confidence": 0.92,
                "token_id": 2
            },
            {
                "text": "Content",
                "bbox": [10, 40, 80, 60],
                "confidence": 0.88,
                "token_id": 3
            }
        ],
        "page_bbox": [0, 0, 224, 224],
        "processing_time": 0.5,
        "model_name": "test_ocr"
    }


@pytest.fixture
def sample_table_data() -> Dict[str, Any]:
    """Sample table detection results."""
    return {
        "tables": [
            {
                "bbox": [20, 80, 200, 150],
                "confidence": 0.9,
                "rows": [
                    {
                        "bbox": [20, 80, 200, 100],
                        "cells": [
                            {"text": "Header 1", "bbox": [20, 80, 110, 100], "confidence": 0.95},
                            {"text": "Header 2", "bbox": [110, 80, 200, 100], "confidence": 0.93}
                        ]
                    },
                    {
                        "bbox": [20, 100, 200, 120],
                        "cells": [
                            {"text": "Data 1", "bbox": [20, 100, 110, 120], "confidence": 0.91},
                            {"text": "Data 2", "bbox": [110, 100, 200, 120], "confidence": 0.89}
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    class MockModel(torch.nn.Module):
        def __init__(self, num_classes=8):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes)
        
        def forward(self, x):
            if isinstance(x, dict):
                # Handle batch input
                batch_size = x["image"].size(0) if "image" in x else 1
                return torch.randn(batch_size, 8)
            else:
                return torch.randn(1, 8)
    
    return MockModel()


@pytest.fixture
def api_client():
    """FastAPI test client."""
    try:
        from api.main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI app not available")


@pytest.fixture
def mock_database_session():
    """Mock database session."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from api.database import Base
        
        # Create in-memory SQLite database
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        yield session
        
        session.close()
        
    except ImportError:
        pytest.skip("Database dependencies not available")


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return {
        "job_id": "test-job-123",
        "status": "pending",
        "file_name": "test_document.pdf",
        "file_size": 1024,
        "mime_type": "application/pdf",
        "confidence_threshold": 0.8
    }


@pytest.fixture
def sample_extraction_results():
    """Sample extraction results."""
    return {
        "rows": [
            {
                "row_id": "row_1",
                "page": 1,
                "region_id": "text_0",
                "bbox": [10, 10, 200, 30],
                "columns": {"text": "Sample extracted text"},
                "provenance": {
                    "file": "test_document.pdf",
                    "page": 1,
                    "bbox": [10, 10, 200, 30],
                    "token_ids": [1, 2, 3],
                    "confidence": 0.92
                },
                "needs_review": False
            }
        ],
        "metadata": {
            "render_type": "digital_pdf",
            "total_pages": 1,
            "confidence_threshold": 0.8
        },
        "confidence_score": 0.92,
        "render_type": "digital_pdf"
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    test_env_vars = {
        "DATABASE_URL": "sqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/15",  # Use test database
        "STORAGE_TYPE": "local",
        "LOCAL_STORAGE_PATH": "/tmp/curioscan_test",
        "API_KEY_REQUIRED": "false",
        "DEBUG": "true"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def mock_celery_task():
    """Mock Celery task for testing."""
    class MockTask:
        def __init__(self, task_id="test-task-123"):
            self.id = task_id
            self.state = "PENDING"
            self.result = None
        
        def get(self, timeout=None):
            return self.result
        
        def ready(self):
            return self.state in ["SUCCESS", "FAILURE"]
        
        def successful(self):
            return self.state == "SUCCESS"
        
        def failed(self):
            return self.state == "FAILURE"
    
    return MockTask()


# Test data generators

def generate_test_dataset(temp_dir: Path, num_samples: int = 10) -> Path:
    """Generate a test dataset."""
    dataset_dir = temp_dir / "test_dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    # Create sample images
    for i in range(num_samples):
        image = Image.new('RGB', (224, 224), color=(i * 25 % 255, 100, 150))
        image.save(dataset_dir / f"sample_{i}.png")
    
    # Create annotations
    annotations = []
    for i in range(num_samples):
        annotations.append({
            "image_path": f"sample_{i}.png",
            "label": i % 3,  # 3 classes
            "metadata": {
                "file_size": 1024,
                "width": 224,
                "height": 224
            }
        })
    
    with open(dataset_dir / "train_annotations.json", 'w') as f:
        json.dump(annotations, f)
    
    return dataset_dir


# Pytest markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.slow = pytest.mark.slow
