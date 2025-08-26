# API Reference

CurioScan exposes a RESTful API for document processing, status tracking, and result retrieval.

## Authentication

All API endpoints require authentication using API keys passed in the `Authorization` header:

```
Authorization: Bearer <your-api-key>
```

Contact your administrator to obtain an API key.

## Endpoints

### Document Processing

#### Upload Document

```
POST /upload
```

Upload a document for processing.

**Request Body:**
- `file`: The document file (PDF, DOCX, image)
- `options`: (Optional) Processing options JSON object

**Processing Options:**
```json
{
  "ocr_engine": "default|advanced|fast",
  "extract_tables": true|false,
  "confidence_threshold": 0.7,
  "language": "eng|fra|deu|...",
  "webhook_url": "https://your-callback-url.com/endpoint"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2025-08-26T14:30:00Z"
}
```

#### Check Processing Status

```
GET /status/{job_id}
```

Get the status of a processing job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing|completed|failed",
  "progress": 65,
  "stage": "text_extraction",
  "created_at": "2025-08-26T14:30:00Z",
  "updated_at": "2025-08-26T14:31:05Z",
  "error": null,
  "preview_url": "https://..."
}
```

### Results

#### Get Processing Results

```
GET /result/{job_id}
```

Get the processing results for a job.

**Query Parameters:**
- `format`: `json|csv|xlsx` (default: json)
- `include_confidence`: `true|false` (default: false)

**Response:**
For JSON format:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "document": {
    "type": "invoice",
    "pages": 3,
    "content": [...],
    "metadata": {...},
    "tables": [...]
  }
}
```

For CSV/XLSX format: Returns a downloadable file.

#### Get Document Preview

```
GET /preview/{job_id}/{page_num}
```

Get a preview image of a processed page.

**Query Parameters:**
- `highlight`: `text|tables|regions` (optional)
- `resolution`: `low|medium|high` (default: medium)

**Response:** PNG image file

### Review System

#### Get Review Items

```
GET /review/{job_id}
```

Get items that need human review.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "review_items": [
    {
      "id": "item-001",
      "page": 1,
      "bbox": [100, 200, 300, 250],
      "text": "Detected text",
      "confidence": 0.65,
      "type": "text|table_cell|amount|date",
      "alternatives": ["Alt 1", "Alt 2"]
    },
    ...
  ]
}
```

#### Submit Review Results

```
POST /review/{job_id}
```

Submit human review corrections.

**Request Body:**
```json
{
  "corrections": [
    {
      "id": "item-001",
      "corrected_text": "Correct text",
      "reviewed_by": "user@example.com"
    },
    ...
  ]
}
```

**Response:**
```json
{
  "success": true,
  "updated_items": 5
}
```

### Webhooks

#### Register Webhook

```
POST /webhooks/register
```

Register a callback URL for job status updates.

**Request Body:**
```json
{
  "url": "https://your-callback-url.com/endpoint",
  "events": ["job.completed", "job.failed"],
  "secret": "your-webhook-secret"
}
```

**Response:**
```json
{
  "webhook_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "active"
}
```

#### List Webhooks

```
GET /webhooks
```

List all registered webhooks.

**Response:**
```json
{
  "webhooks": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "url": "https://your-callback-url.com/endpoint",
      "events": ["job.completed", "job.failed"],
      "created_at": "2025-08-26T14:30:00Z"
    },
    ...
  ]
}
```

### Model Training

#### Trigger Retraining

```
POST /retrain-trigger
```

Trigger model retraining with reviewed data.

**Request Body:**
```json
{
  "model_type": "text|table|layout",
  "include_jobs": ["job-id-1", "job-id-2"],
  "training_params": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

**Response:**
```json
{
  "training_job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "estimated_completion": "2025-08-26T16:30:00Z"
}
```

## Error Handling

All API errors return appropriate HTTP status codes along with error details:

```json
{
  "error": {
    "code": "invalid_document",
    "message": "The uploaded document is corrupted or in an unsupported format",
    "details": "..."
  }
}
```

Common error codes include:
- `invalid_document` - Document format issues
- `processing_error` - Error during document processing
- `not_found` - Job ID not found
- `unauthorized` - Invalid or missing API key
- `rate_limited` - Too many requests
- `internal_error` - Server-side error
