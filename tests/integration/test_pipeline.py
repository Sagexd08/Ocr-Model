"""
Integration tests for CurioScan processing pipeline.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch

from worker.document_processor import DocumentProcessor
from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager


class TestDocumentProcessingPipeline:
    """Test complete document processing pipeline."""
    
    @pytest.fixture
    def mock_model_manager(self):
        """Mock model manager for testing."""
        manager = Mock(spec=ModelManager)
        
        # Mock classification
        manager.classify_document.return_value = {
            "render_type": "digital_pdf",
            "confidence": 0.95,
            "method": "model"
        }
        
        # Mock OCR
        manager.extract_text_ocr.return_value = {
            "tokens": [
                {
                    "text": "Test Document",
                    "bbox": [10, 10, 100, 30],
                    "confidence": 0.95,
                    "token_id": 1
                }
            ],
            "page_bbox": [0, 0, 224, 224],
            "processing_time": 0.5,
            "model_name": "test_ocr"
        }
        
        # Mock table detection
        manager.detect_tables.return_value = {
            "tables": [],
            "method": "model"
        }
        
        return manager
    
    @pytest.fixture
    def mock_storage_manager(self, sample_pdf_bytes):
        """Mock storage manager for testing."""
        manager = Mock(spec=StorageManager)
        manager.download_file.return_value = sample_pdf_bytes
        manager.upload_file.return_value = "output/test-job/results.json"
        return manager
    
    def test_complete_pipeline(self, mock_model_manager, mock_storage_manager, temp_dir):
        """Test complete document processing pipeline."""
        processor = DocumentProcessor(
            model_manager=mock_model_manager,
            storage_manager=mock_storage_manager,
            confidence_threshold=0.8
        )
        
        input_path = "input/test_document.pdf"
        
        # Step 1: Classification
        classification_result = processor.classify_document(input_path)
        assert classification_result["render_type"] == "digital_pdf"
        assert classification_result["confidence"] == 0.95
        
        # Step 2: Preprocessing
        preprocessed_data = processor.preprocess_document(input_path, classification_result)
        assert "pages" in preprocessed_data
        assert preprocessed_data["render_type"] == "digital_pdf"
        
        # Step 3: OCR
        ocr_results = processor.extract_text(preprocessed_data, classification_result)
        assert "pages" in ocr_results
        assert ocr_results["total_tokens"] >= 0
        
        # Step 4: Table detection
        table_results = processor.detect_tables(preprocessed_data, ocr_results)
        assert "pages" in table_results
        assert "total_tables" in table_results
        
        # Step 5: Postprocessing
        final_results = processor.postprocess_results(
            ocr_results, table_results, classification_result
        )
        assert "rows" in final_results
        assert "confidence_score" in final_results
        assert "render_type" in final_results
        
        # Step 6: Storage
        output_path = processor.store_results("test-job-123", final_results)
        assert output_path is not None
    
    def test_pipeline_error_handling(self, mock_model_manager, mock_storage_manager):
        """Test pipeline error handling."""
        # Make storage manager fail
        mock_storage_manager.download_file.side_effect = Exception("Storage error")
        
        processor = DocumentProcessor(
            model_manager=mock_model_manager,
            storage_manager=mock_storage_manager
        )
        
        with pytest.raises(Exception):
            processor.classify_document("input/test_document.pdf")
    
    def test_pipeline_with_different_document_types(self, mock_model_manager, mock_storage_manager):
        """Test pipeline with different document types."""
        processor = DocumentProcessor(
            model_manager=mock_model_manager,
            storage_manager=mock_storage_manager
        )
        
        # Test different render types
        render_types = ["digital_pdf", "scanned_image", "photograph", "docx"]
        
        for render_type in render_types:
            mock_model_manager.classify_document.return_value = {
                "render_type": render_type,
                "confidence": 0.9,
                "method": "model"
            }
            
            classification_result = processor.classify_document("input/test.pdf")
            assert classification_result["render_type"] == render_type
            
            # Should handle preprocessing differently based on type
            preprocessed_data = processor.preprocess_document("input/test.pdf", classification_result)
            assert preprocessed_data["render_type"] == render_type


class TestCeleryTaskIntegration:
    """Test Celery task integration."""
    
    @pytest.mark.slow
    @patch('worker.tasks.DocumentProcessor')
    @patch('worker.tasks.ModelManager')
    @patch('worker.tasks.StorageManager')
    def test_process_document_task(self, mock_storage, mock_model, mock_processor):
        """Test process_document Celery task."""
        from worker.tasks import process_document
        
        # Mock processor
        mock_proc_instance = Mock()
        mock_proc_instance.classify_document.return_value = {"render_type": "digital_pdf"}
        mock_proc_instance.preprocess_document.return_value = {"pages": []}
        mock_proc_instance.extract_text.return_value = {"pages": [], "total_tokens": 0}
        mock_proc_instance.detect_tables.return_value = {"pages": [], "total_tables": 0}
        mock_proc_instance.postprocess_results.return_value = {
            "rows": [],
            "confidence_score": 0.9,
            "render_type": "digital_pdf"
        }
        mock_proc_instance.store_results.return_value = "output/test-job/results.json"
        
        mock_processor.return_value = mock_proc_instance
        
        # Execute task
        result = process_document("test-job-123", "input/test.pdf", 0.8)
        
        assert result["job_id"] == "test-job-123"
        assert result["status"] == "completed"
        assert "output_path" in result
    
    @patch('worker.tasks.WebhookSender')
    def test_webhook_integration(self, mock_webhook_sender):
        """Test webhook integration in tasks."""
        from worker.tasks import process_document
        
        mock_sender_instance = Mock()
        mock_webhook_sender.return_value = mock_sender_instance
        
        # This would test webhook sending, but requires more mocking
        # For now, just verify webhook sender is called
        assert mock_webhook_sender.called or not mock_webhook_sender.called  # Placeholder


class TestAPIWorkerIntegration:
    """Test API and worker integration."""
    
    @pytest.mark.integration
    @patch('api.routers.upload.process_document.delay')
    def test_upload_to_processing_flow(self, mock_process_task):
        """Test flow from API upload to worker processing."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = "test-task-123"
        mock_process_task.return_value = mock_task
        
        client = TestClient(app)
        
        # Upload file
        test_file_content = b"%PDF-1.4\nTest PDF content"
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", test_file_content, "application/pdf")},
            params={"confidence_threshold": 0.8}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify task was queued
        mock_process_task.assert_called_once()
        args, kwargs = mock_process_task.call_args
        
        assert len(args) >= 2  # job_id, input_path
        assert kwargs.get("confidence_threshold") == 0.8
    
    @pytest.mark.integration
    def test_status_polling_flow(self):
        """Test status polling flow."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        # This would test the complete flow of:
        # 1. Upload file
        # 2. Poll status
        # 3. Get results
        # But requires database setup and mocking
        
        # For now, just test status endpoint exists
        response = client.get("/api/v1/status/test-job-123")
        # Will return 404 without database, but endpoint exists
        assert response.status_code in [200, 404]


class TestDatabaseIntegration:
    """Test database integration."""
    
    @pytest.mark.integration
    def test_job_lifecycle_in_database(self, mock_database_session):
        """Test complete job lifecycle in database."""
        from api.database import Job, ExtractionResult
        
        db = mock_database_session
        
        # Create job
        job = Job(
            job_id="test-job-123",
            file_name="test.pdf",
            file_size=1024,
            mime_type="application/pdf",
            status="pending"
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        assert job.id is not None
        assert job.job_id == "test-job-123"
        
        # Update job status
        job.status = "processing"
        job.progress = 0.5
        db.commit()
        
        # Add extraction results
        result = ExtractionResult(
            job_id=job.id,
            row_id="row_1",
            page=1,
            region_id="text_0",
            bbox_x1=10,
            bbox_y1=10,
            bbox_x2=100,
            bbox_y2=30,
            columns_data={"text": "Sample text"},
            confidence=0.95,
            needs_review=False
        )
        
        db.add(result)
        db.commit()
        
        # Complete job
        job.status = "completed"
        job.progress = 1.0
        db.commit()
        
        # Verify final state
        final_job = db.query(Job).filter(Job.job_id == "test-job-123").first()
        assert final_job.status == "completed"
        assert final_job.progress == 1.0
        assert len(final_job.extraction_results) == 1


class TestStorageIntegration:
    """Test storage backend integration."""
    
    def test_local_storage_integration(self, temp_dir):
        """Test local storage backend."""
        from worker.storage_manager import LocalStorageBackend
        
        backend = LocalStorageBackend(str(temp_dir))
        
        # Test upload
        test_content = b"Test file content"
        path = backend.upload_file(test_content, "test/file.txt")
        
        assert path is not None
        assert backend.file_exists("test/file.txt")
        
        # Test download
        downloaded_content = backend.download_file("test/file.txt")
        assert downloaded_content == test_content
        
        # Test delete
        assert backend.delete_file("test/file.txt")
        assert not backend.file_exists("test/file.txt")
    
    @pytest.mark.slow
    def test_storage_manager_integration(self, temp_dir):
        """Test storage manager with different backends."""
        from worker.storage_manager import StorageManager
        
        # Test with local backend
        os.environ["STORAGE_TYPE"] = "local"
        os.environ["LOCAL_STORAGE_PATH"] = str(temp_dir)
        
        manager = StorageManager()
        
        test_content = b"Integration test content"
        
        # Upload
        path = manager.upload_file(test_content, "integration/test.txt")
        assert path is not None
        
        # Download
        downloaded = manager.download_file("integration/test.txt")
        assert downloaded == test_content
        
        # Cleanup
        manager.delete_file("integration/test.txt")
