# CurioScan Architecture

## System Overview

CurioScan is designed as a modular, scalable OCR system for document processing with the following high-level architecture:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    API      │─────▶│    Worker   │─────▶│   Storage   │
│  (FastAPI)  │      │  (Celery)   │      │   (MinIO)   │
└─────────────┘      └─────────────┘      └─────────────┘
       ▲                    │                    │
       │                    │                    │
       │                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Review UI   │      │   Models    │      │  Database   │
│  (Streamlit)│◀─────│  (PyTorch)  │◀─────│  (SQLite)   │
└─────────────┘      └─────────────┘      └─────────────┘
```

## Components

### API Layer (FastAPI)

The API layer exposes RESTful endpoints for:
- Document upload and processing
- Job status checking
- Result retrieval and export
- Review handling
- Webhook management
- Model retraining triggers

### Worker System (Celery)

The worker system handles the computational tasks:
- Document preprocessing
- OCR model inference
- Layout analysis and text extraction
- Table detection and processing
- Post-processing and normalization
- Result persistence

### Processing Pipeline

The document processing pipeline consists of the following stages:

1. **Document Classification**
   - Identify document type (invoice, receipt, form, etc.)
   - Classify render type (digital PDF, scanned image, photograph)

2. **Preprocessing**
   - Image enhancement
   - Noise reduction
   - Deskewing and rotation correction
   - Color normalization and binarization

3. **Layout Analysis**
   - Page segmentation
   - Region classification (text, table, figure, etc.)
   - Reading order determination

4. **Text Recognition**
   - Text extraction from digital PDFs
   - OCR for scanned/image documents
   - Handwriting recognition where applicable

5. **Table Processing**
   - Table detection and boundary identification
   - Table structure analysis
   - Cell content extraction and normalization

6. **Post-processing**
   - Text normalization
   - Entity extraction
   - Confidence scoring
   - Quality analysis

7. **Export**
   - Structured data generation (JSON, CSV, XLSX)
   - Metadata and confidence inclusion
   - Provenance tracking

## Data Flow

1. User uploads document via API
2. API validates input and creates processing job
3. Worker picks up job and processes document through pipeline
4. Results stored in MinIO object storage
5. Low-confidence items flagged for human review
6. Reviewers correct data via UI
7. Corrections stored and used for model improvement
8. Finalized results available for export

## Scalability & Performance

The system is designed to scale horizontally with the following considerations:

- Docker containers for consistent deployments
- Celery workers can be scaled based on demand
- Batched processing for larger documents
- Caching layer for frequently accessed results
- Kubernetes deployment for production environments

## Security Considerations

- Authentication and authorization for all API endpoints
- Input validation and sanitization
- Secure storage of sensitive document data
- Rate limiting to prevent abuse
- Data encryption at rest and in transit
