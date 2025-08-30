# CurioScan

CurioScan is a production-grade OCR system for ingesting, understanding, and exporting documents.

## Features

- **Ingestion**: Handles PDFs (digital & scanned), DOCX, TIFF/JPEG/PNG, and photographed pages.
- **Classification**: Classifies render type (digital_pdf, scanned_image, etc.) for specialized processing.
- **Export**: Produces 1:1 faithful exports in CSV, XLSX, and JSON with detailed provenance.
- **Human-in-the-loop**: Surfaces low-confidence items for review and feeds corrections back into the system.
- **Scalable & Deployable**: Built with Docker, docker-compose, Helm, and Kubernetes for scalability.

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Local Demo

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/curioscan.git
    cd curioscan
    ```

2.  **Run the demo:**

    ```bash
    docker-compose up
    ```

    This will start the following services:

    -   `api`: FastAPI backend
    -   `worker`: Celery worker
    -   `redis`: Message broker
    -   `minio`: S3-compatible object storage
    -   `streamlit`: Streamlit demo UI
    -   `frontend`: React-based review UI

3.  **Access the demo:**

    -   **Streamlit UI**: http://localhost:8501
    -   **Review UI**: http://localhost:3000/review/{job_id}
    -   **API**: http://localhost:8000/docs

## Usage

### Training

To train a model, run the following command:

```bash
python training/train.py --config configs/demo.yaml
```

### Evaluation

To evaluate a model, run the following command:

```bash
python evaluation/evaluate.py --dataset demo_holdout
```

### End-to-End Demo

Run the end-to-end demo script:

```bash
./demo_process.sh
```

## Project Structure

```
├── api
├── worker
├── models
├── training
├── streamlit_demo
├── frontend
├── deploy
├── tests
├── # CurioScan OCR Processing System

A scalable, production-ready OCR-powered intelligent document processing system capable of parsing raw PDFs, Word documents, scanned images, and multi-page documents with high accuracy.

## Features

- **Multi-Model OCR Processing**: Combines Tesseract, PaddleOCR, and optimized ONNX models for high accuracy
- **Intelligent Document Classification**: Automatically classifies document types
- **Form Field Detection**: Extracts structured form data including checkboxes and input fields
- **Table Detection and Extraction**: Identifies and structures tabular data
- **Hybrid PDF Processing**: Handles both native (digital) PDF text and scanned PDF pages
- **Layout Analysis**: Preserves document layout and structure
- **Multi-Format Export**: Export to JSON, CSV, Excel, and plain text
- **Asynchronous Processing**: Celery-based distributed task processing
- **API-First Design**: RESTful API for document uploading and retrieval

## System Architecture

The system consists of the following main components:

- **API Server**: FastAPI-based REST API for document upload and retrieval
- **Worker**: Celery workers that process documents asynchronously
- **Storage Manager**: Handles document storage and retrieval
- **Model Manager**: Manages ML models and their lifecycle
- **Processing Pipeline**: Configurable pipeline of document processors

### Processing Pipeline

The document processing pipeline consists of these stages:

1. **Document Type Detection**: Determine document type (PDF, image, Word, etc.)
2. **PDF Processing**: Extract text from PDFs, handle hybrid content
3. **OCR Processing**: Extract text from images using ONNX-optimized OCR
4. **Layout Analysis**: Analyze document layout and structure
5. **Table Detection**: Identify and extract tables
6. **Form Field Extraction**: Extract form fields and values
7. **Post-Processing**: Clean and normalize text
8. **Export**: Package results in requested formats

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL
- Redis
- Docker (optional)

### Local Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the database:

```bash
alembic upgrade head
```

4. Start the API server:

```bash
uvicorn api.main:app --reload
```

5. Start the Celery worker:

```bash
celery -A worker.celery_app worker --loglevel=info
```

### Docker Setup

1. Build and start the containers:

```bash
docker-compose up -d
```

## Usage

### API Endpoints

- `POST /api/upload`: Upload a document for processing
- `GET /api/status/{job_id}`: Check processing status
- `GET /api/results/{job_id}`: Get processing results
- `GET /api/download/{job_id}`: Download processed document

### Command Line

For quick testing, use the test script:

```bash
python test_processor.py path/to/document.pdf
```

## Development

### Adding New Processors

1. Create a new processor class in `worker/pipeline/processors/`
2. Implement the `process` method
3. Register the processor in `worker/pipeline/pipeline_builder.py`

### Running Tests

```bash
pytest tests
```

## License

Proprietary - All Rights Reserved
└── ...
```

## API Contract

- `POST /upload` -> `{ job_id }`
- `GET /status/{job_id}` -> `{ status, progress, preview }`
- `GET /result/{job_id}` -> downloadable CSV/XLSX/JSON + provenance
- `POST /webhooks/register` -> register callback
- `GET /review/{job_id}` -> paginated review payload for UI
- `POST /review/{job_id}` -> save corrected data
- `POST /retrain-trigger` -> trigger retraining with accepted corrections

## LLM Usage

You are a strict normalizer. Input: OCR tokens with bbox and confidence. Output: JSON matching the schema. Do not invent missing values. If unsure, set needs_review=true. Output JSON only.


## Local Run (no Docker)

- Create venv and install deps:
  - python -m venv .venv
  - .venv\\Scripts\\pip install -r requirements.txt
- Initialize SQLite (auto-created on first run):
  - setx CURIO_TEST_MODE 0
  - setx DATABASE_URL sqlite+pysqlite:///./curioscan.db
- Start API:
  - .venv\\Scripts\\uvicorn api.main:app --reload
- Health check:
  - curl http://localhost:8000/health
- Upload test:
  - curl -F "file=@tests/data/sample.pdf" http://localhost:8000/upload
