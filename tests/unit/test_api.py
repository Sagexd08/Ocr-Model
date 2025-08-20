"""
Unit tests for CurioScan API.
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Skip if FastAPI not available
pytest.importorskip("fastapi")

from api.main import app
from api.models import JobStatus


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health endpoint returns 200."""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_check_includes_timestamp(self):
        """Test health check includes timestamp."""
        client = TestClient(app)
        response = client.get("/health")
        
        data = response.json()
        assert "timestamp" in data
        assert "version" in data


class TestUploadEndpoint:
    """Test file upload endpoint."""
    
    @patch('api.routers.upload.process_document.delay')
    def test_upload_pdf_file(self, mock_process):
        """Test uploading a PDF file."""
        mock_process.return_value = Mock(id="test-task-123")
        
        client = TestClient(app)
        
        # Create test file
        test_file_content = b"%PDF-1.4\nTest PDF content"
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", test_file_content, "application/pdf")},
            params={"confidence_threshold": 0.8}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "uploaded"
    
    def test_upload_invalid_file_type(self):
        """Test uploading invalid file type."""
        client = TestClient(app)
        
        test_file_content = b"Invalid file content"
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", test_file_content, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_upload_no_file(self):
        """Test upload without file."""
        client = TestClient(app)
        
        response = client.post("/api/v1/upload")
        
        assert response.status_code == 422  # Validation error
    
    def test_upload_large_file(self):
        """Test uploading file that's too large."""
        client = TestClient(app)
        
        # Create large file content (simulate)
        large_content = b"x" * (100 * 1024 * 1024)  # 100MB
        
        response = client.post(
            "/api/v1/upload",
            files={"file": ("large.pdf", large_content, "application/pdf")}
        )
        
        # Should reject large files
        assert response.status_code in [413, 400]


class TestStatusEndpoint:
    """Test job status endpoint."""
    
    @patch('api.routers.status.get_db')
    def test_get_job_status_found(self, mock_get_db):
        """Test getting status of existing job."""
        # Mock database
        mock_db = Mock()
        mock_job = Mock()
        mock_job.job_id = "test-job-123"
        mock_job.status = JobStatus.PROCESSING
        mock_job.progress = 0.5
        mock_job.created_at = "2023-01-01T00:00:00"
        mock_job.updated_at = "2023-01-01T00:30:00"
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_job
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        response = client.get("/api/v1/status/test-job-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == JobStatus.PROCESSING
        assert data["progress"] == 0.5
    
    @patch('api.routers.status.get_db')
    def test_get_job_status_not_found(self, mock_get_db):
        """Test getting status of non-existent job."""
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        response = client.get("/api/v1/status/nonexistent-job")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data


class TestJobsEndpoint:
    """Test jobs listing endpoint."""
    
    @patch('api.routers.status.get_db')
    def test_list_jobs(self, mock_get_db):
        """Test listing jobs."""
        # Mock database
        mock_db = Mock()
        mock_jobs = [
            Mock(
                job_id="job-1",
                status=JobStatus.COMPLETED,
                file_name="test1.pdf",
                created_at="2023-01-01T00:00:00"
            ),
            Mock(
                job_id="job-2", 
                status=JobStatus.PROCESSING,
                file_name="test2.pdf",
                created_at="2023-01-01T01:00:00"
            )
        ]
        
        mock_db.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_jobs
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        response = client.get("/api/v1/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["job_id"] == "job-1"
        assert data[1]["job_id"] == "job-2"
    
    @patch('api.routers.status.get_db')
    def test_list_jobs_with_filter(self, mock_get_db):
        """Test listing jobs with status filter."""
        mock_db = Mock()
        mock_jobs = [
            Mock(
                job_id="job-1",
                status=JobStatus.COMPLETED,
                file_name="test1.pdf"
            )
        ]
        
        mock_db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_jobs
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        response = client.get("/api/v1/jobs?status=completed")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == JobStatus.COMPLETED


class TestResultsEndpoint:
    """Test results download endpoint."""
    
    @patch('api.routers.results.get_db')
    def test_download_results_json(self, mock_get_db):
        """Test downloading results in JSON format."""
        # Mock database
        mock_db = Mock()
        mock_job = Mock()
        mock_job.job_id = "test-job-123"
        mock_job.status = JobStatus.COMPLETED
        mock_job.extraction_results = []
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_job
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        response = client.get("/api/v1/result/test-job-123?format=json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    @patch('api.routers.results.get_db')
    def test_download_results_job_not_found(self, mock_get_db):
        """Test downloading results for non-existent job."""
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        response = client.get("/api/v1/result/nonexistent-job")
        
        assert response.status_code == 404
    
    @patch('api.routers.results.get_db')
    def test_download_results_job_not_completed(self, mock_get_db):
        """Test downloading results for incomplete job."""
        mock_db = Mock()
        mock_job = Mock()
        mock_job.status = JobStatus.PROCESSING
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_job
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        response = client.get("/api/v1/result/test-job-123")
        
        assert response.status_code == 400


class TestWebhooksEndpoint:
    """Test webhook management endpoints."""
    
    @patch('api.routers.webhooks.get_db')
    def test_register_webhook(self, mock_get_db):
        """Test registering a webhook."""
        mock_db = Mock()
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["job.completed", "job.failed"],
            "secret": "test-secret"
        }
        
        response = client.post("/api/v1/webhooks/register", json=webhook_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "webhook_id" in data
        assert data["url"] == webhook_data["url"]
        assert data["events"] == webhook_data["events"]
    
    def test_register_webhook_invalid_url(self):
        """Test registering webhook with invalid URL."""
        client = TestClient(app)
        
        webhook_data = {
            "url": "invalid-url",
            "events": ["job.completed"]
        }
        
        response = client.post("/api/v1/webhooks/register", json=webhook_data)
        
        assert response.status_code == 400
    
    def test_register_webhook_invalid_events(self):
        """Test registering webhook with invalid events."""
        client = TestClient(app)
        
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["invalid.event"]
        }
        
        response = client.post("/api/v1/webhooks/register", json=webhook_data)
        
        assert response.status_code == 400


class TestReviewEndpoint:
    """Test human review endpoints."""
    
    @patch('api.routers.review.get_db')
    def test_get_review_items(self, mock_get_db):
        """Test getting review items."""
        mock_db = Mock()
        mock_job = Mock()
        mock_job.id = 1
        mock_job.job_id = "test-job-123"
        
        mock_review_items = []
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_job
        mock_db.query.return_value.filter.return_value.count.return_value = 0
        mock_db.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = mock_review_items
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        response = client.get("/api/v1/review/test-job-123")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total_count" in data
        assert "page" in data
    
    @patch('api.routers.review.get_db')
    def test_update_review_items(self, mock_get_db):
        """Test updating review items."""
        mock_db = Mock()
        mock_review_item = Mock()
        mock_review_item.item_id = "item-123"
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_review_item
        mock_get_db.return_value = mock_db
        
        client = TestClient(app)
        
        updates = [
            {
                "item_id": "item-123",
                "action": "accept"
            }
        ]
        
        response = client.post("/api/v1/review/update", json=updates)
        
        assert response.status_code == 200
        data = response.json()
        assert "updated_items" in data


class TestRetrainEndpoint:
    """Test model retraining endpoints."""
    
    def test_trigger_retraining_dry_run(self):
        """Test triggering retraining in dry run mode."""
        client = TestClient(app)
        
        retrain_data = {
            "model_type": "renderer_classifier",
            "dataset_path": "data/corrected_extractions",
            "config_overrides": {},
            "dry_run": True
        }
        
        response = client.post("/api/v1/retrain-trigger", json=retrain_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "retrain_job_id" in data
        assert data["status"] == "dry_run_completed"
    
    def test_trigger_retraining_invalid_model_type(self):
        """Test triggering retraining with invalid model type."""
        client = TestClient(app)
        
        retrain_data = {
            "model_type": "invalid_model",
            "dataset_path": "data/test",
            "config_overrides": {},
            "dry_run": True
        }
        
        response = client.post("/api/v1/retrain-trigger", json=retrain_data)
        
        assert response.status_code == 400


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_404_endpoint(self):
        """Test 404 for non-existent endpoint."""
        client = TestClient(app)
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test 405 for wrong HTTP method."""
        client = TestClient(app)
        response = client.delete("/health")
        
        assert response.status_code == 405
    
    def test_validation_error(self):
        """Test validation error handling."""
        client = TestClient(app)
        
        # Send invalid JSON
        response = client.post(
            "/api/v1/webhooks/register",
            json={"invalid": "data"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
