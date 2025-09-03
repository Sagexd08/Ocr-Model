# 🚀 API Documentation

## Overview

The Enterprise OCR Processing System provides a comprehensive RESTful API for document processing, job management, and system monitoring.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API uses basic authentication. Future versions will support OAuth2 and API keys.

## Endpoints

### 📤 Document Upload

#### POST `/upload`

Upload and process a document.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf" \
     -F "job_id=my-job-123" \
     -F "profile=performance" \
     -F "mode=advanced" \
     -F "max_pages=5"
```

**Parameters:**
- `file` (required): Document file
- `job_id` (optional): Custom job identifier
- `profile` (optional): Processing profile (`performance`, `quality`, `balanced`)
- `mode` (optional): Processing mode (`advanced`, `standard`, `fast`)
- `max_pages` (optional): Maximum pages to process

**Response:**
```json
{
  "job_id": "my-job-123",
  "status": "processing",
  "message": "Document uploaded successfully",
  "estimated_completion": "2025-09-03T10:35:00Z"
}
```

### 📊 Job Management

#### GET `/jobs/{job_id}`

Get job status and results.

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/jobs/my-job-123"
```

**Response:**
```json
{
  "job_id": "my-job-123",
  "status": "completed",
  "progress": 100,
  "summary": {
    "page_count": 5,
    "word_count": 1250,
    "confidence_avg": 0.95,
    "processing_time": 45.2
  },
  "created_at": "2025-09-03T10:30:00Z",
  "completed_at": "2025-09-03T10:30:45Z",
  "result_url": "/api/v1/jobs/my-job-123/download"
}
```

#### GET `/jobs/{job_id}/download`

Download processing results.

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/jobs/my-job-123/download" \
     -H "Accept: application/json"
```

**Response:** JSON file with complete processing results

#### DELETE `/jobs/{job_id}`

Cancel or delete a job.

**Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/jobs/my-job-123"
```

### 📋 System Information

#### GET `/health`

System health check.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime": 3600,
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 68.5,
    "disk_usage": 23.1
  }
}
```

#### GET `/stats`

System statistics.

**Response:**
```json
{
  "total_jobs": 1250,
  "completed_jobs": 1200,
  "failed_jobs": 15,
  "average_processing_time": 32.5,
  "total_pages_processed": 15000,
  "total_words_extracted": 2500000
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "PROCESSING_FAILED",
    "message": "Document processing failed",
    "details": "OCR engine encountered an error",
    "timestamp": "2025-09-03T10:30:00Z"
  }
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_FILE` | Unsupported file format | 400 |
| `FILE_TOO_LARGE` | File exceeds size limit | 413 |
| `PROCESSING_FAILED` | Processing error | 500 |
| `JOB_NOT_FOUND` | Job ID not found | 404 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |

## Rate Limiting

- **Upload**: 10 requests per minute
- **Status Check**: 100 requests per minute
- **Download**: 50 requests per minute

## SDK Examples

### Python SDK

```python
import requests

class OCRClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def upload_document(self, file_path, profile="balanced"):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'profile': profile}
            response = requests.post(f"{self.base_url}/upload", files=files, data=data)
            return response.json()
    
    def get_job_status(self, job_id):
        response = requests.get(f"{self.base_url}/jobs/{job_id}")
        return response.json()
    
    def download_results(self, job_id):
        response = requests.get(f"{self.base_url}/jobs/{job_id}/download")
        return response.json()

# Usage
client = OCRClient()
result = client.upload_document("document.pdf", profile="performance")
job_id = result["job_id"]

# Poll for completion
import time
while True:
    status = client.get_job_status(job_id)
    if status["status"] == "completed":
        results = client.download_results(job_id)
        break
    time.sleep(5)
```

### JavaScript SDK

```javascript
class OCRClient {
    constructor(baseUrl = 'http://localhost:8000/api/v1') {
        this.baseUrl = baseUrl;
    }
    
    async uploadDocument(file, profile = 'balanced') {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('profile', profile);
        
        const response = await fetch(`${this.baseUrl}/upload`, {
            method: 'POST',
            body: formData
        });
        
        return response.json();
    }
    
    async getJobStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}`);
        return response.json();
    }
    
    async downloadResults(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}/download`);
        return response.json();
    }
}

// Usage
const client = new OCRClient();
const fileInput = document.getElementById('file-input');
const file = fileInput.files[0];

client.uploadDocument(file, 'performance')
    .then(result => {
        const jobId = result.job_id;
        // Poll for completion
        const checkStatus = () => {
            client.getJobStatus(jobId)
                .then(status => {
                    if (status.status === 'completed') {
                        return client.downloadResults(jobId);
                    } else {
                        setTimeout(checkStatus, 5000);
                    }
                })
                .then(results => {
                    console.log('Processing complete:', results);
                });
        };
        checkStatus();
    });
```

## Webhooks

Configure webhooks to receive notifications when jobs complete.

### Webhook Configuration

```bash
curl -X POST "http://localhost:8000/api/v1/webhooks" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://your-app.com/webhook",
       "events": ["job.completed", "job.failed"],
       "secret": "your-webhook-secret"
     }'
```

### Webhook Payload

```json
{
  "event": "job.completed",
  "job_id": "my-job-123",
  "timestamp": "2025-09-03T10:30:45Z",
  "data": {
    "status": "completed",
    "summary": {
      "page_count": 5,
      "word_count": 1250,
      "confidence_avg": 0.95
    }
  }
}
```
