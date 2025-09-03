# 🔍 Enterprise OCR Processing System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7%2B-green.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Sagexd08/Ocr-Model/graphs/commit-activity)

> **Enterprise-grade OCR document processing system with advanced analytics, real-time monitoring, and comprehensive export capabilities.**

## 🌟 Key Features

### 🚀 **Advanced OCR Engine**
- **State-of-the-art Accuracy**: PaddleOCR integration with 98%+ text recognition accuracy
- **Multi-format Support**: PDF, PNG, JPG, JPEG, TIFF, BMP, WebP, DOCX
- **Intelligent Processing**: Automatic document classification and metadata extraction
- **Table Detection**: Advanced table extraction and structured data analysis
- **Batch Processing**: Queue-based processing with real-time progress tracking

### 🎛️ **Enterprise Web Interface**
- **Advanced Analytics Dashboard**: Real-time confidence scoring and quality assessment
- **Interactive Visualizations**: Plotly-powered charts and performance metrics
- **System Monitoring**: Comprehensive processing history and trend analysis
- **Professional UI**: Modern gradient styling with responsive design
- **Export Options**: Multiple formats (TXT, CSV, JSON, Markdown Reports)

### ⚡ **Performance Profiles**
- **Performance Mode**: Optimized for speed (~5-10 seconds per page)
- **Quality Mode**: Maximum accuracy with detailed analysis
- **Balanced Mode**: Optimal speed/accuracy ratio for production use

### 🔧 **Advanced Features**
- **Real-time Processing**: Live status updates and progress tracking
- **Error Recovery**: Robust error handling and system resilience
- **Caching System**: Intelligent caching for improved performance
- **API Integration**: RESTful API for programmatic access
- **CLI Tools**: Command-line interface for automation and batch processing

## 📸 System Screenshots

### 🎛️ Main Dashboard
![Enterprise Dashboard](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Enterprise+OCR+Dashboard)
*Advanced analytics dashboard with real-time metrics and processing history*

### 📊 Document Processing
![Document Processing](https://via.placeholder.com/800x400/2ca02c/ffffff?text=Document+Processing+Interface)
*Intuitive document upload and processing with live progress tracking*

### 📈 Analytics & Insights
![Analytics View](https://via.placeholder.com/800x400/ff7f0e/ffffff?text=Advanced+Analytics+%26+Insights)
*Comprehensive analytics with confidence scoring and quality assessment*

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free disk space
- **Network**: Internet connection for initial model download

### 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sagexd08/Ocr-Model.git
   cd Ocr-Model
   ```

2. **Install dependencies**
   ```bash
   # Core OCR system
   pip install -r requirements.txt

   # Web interface dependencies
   pip install -r requirements_streamlit.txt
   ```

3. **Launch the application**
   ```bash
   # Option 1: Enterprise Launcher (Recommended)
   python launch_advanced_ocr.py

   # Option 2: Direct Streamlit
   streamlit run advanced_ocr_app.py --server.port 8505

   # Option 3: Command Line Processing
   python -m cli.process_pdf "document.pdf" --profile performance
   ```

4. **Access the web interface**
   Open your browser to: **http://localhost:8505**

### 🐳 Docker Deployment (Optional)

```bash
# Build and run with Docker
docker-compose up --build

# Access services:
# - Streamlit UI: http://localhost:8501
# - API: http://localhost:8000/docs
# - Review UI: http://localhost:3000
```

## 📖 Comprehensive Usage Guide

### 🌐 **Web Interface Workflow**

#### 1. **System Initialization**
- Click "🚀 Initialize OCR System" in the sidebar
- Wait for AI models to load (~30-60 seconds)
- Verify system status shows "🟢 Active"

#### 2. **Document Processing**
- Navigate to "📤 Document Processing" tab
- Upload document via drag & drop or file browser
- Configure processing settings:
  - **Profile**: Performance, Quality, or Balanced
  - **Mode**: Advanced, Standard, or Fast
  - **Options**: Table extraction, document classification
- Click "🚀 Process Document"
- Monitor real-time processing status

#### 3. **Results Analysis**
- Switch to "📊 Current Results" tab
- Review extracted text with confidence scores
- Analyze quality metrics and performance data
- Explore token-level details and analytics
- Export results in multiple formats

#### 4. **System Monitoring**
- Use "🎛️ System Dashboard" for comprehensive overview
- Track processing history and performance trends
- Monitor system statistics and health metrics
- Export analytics and generate reports

### 💻 **Command Line Interface**

#### Basic Processing
```bash
# Process a single document
python -m cli.process_pdf "document.pdf"

# Process with specific profile
python -m cli.process_pdf "document.pdf" --profile performance

# Process multiple pages
python -m cli.process_pdf "document.pdf" --max-pages 10
```

#### Advanced Processing
```bash
# Full-featured processing
python -m cli.process_pdf "document.pdf" \
  --profile quality \
  --mode advanced \
  --max-pages 5 \
  --export json \
  --output-dir results/

# Batch processing
python -m cli.process_pdf "folder/*.pdf" \
  --profile performance \
  --mode fast \
  --export csv
```

#### Processing Options
| Option | Description | Values |
|--------|-------------|--------|
| `--profile` | Processing profile | `performance`, `quality`, `balanced` |
| `--mode` | Processing mode | `advanced`, `standard`, `fast` |
| `--max-pages` | Page limit | `1-100` |
| `--export` | Export format | `json`, `csv`, `txt` |
| `--output-dir` | Output directory | Any valid path |

### 🐍 **Python API Integration**

#### Basic Usage
```python
from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager
from worker.document_processor import EnhancedDocumentProcessor

# Initialize system components
model_manager = ModelManager()
storage_manager = StorageManager()
processor = EnhancedDocumentProcessor(model_manager, storage_manager)

# Process document
result = processor.process_document(
    job_id="my_processing_job",
    document_path="path/to/document.pdf",
    params={
        "mode": "advanced",
        "profile": "performance",
        "max_pages": 5
    }
)

# Access results
print(f"Status: {result['status']}")
print(f"Pages processed: {result['summary']['page_count']}")
print(f"Words extracted: {result['summary']['word_count']}")
```

#### Advanced Configuration
```python
# Advanced processing with all options
result = processor.process_document(
    job_id="advanced_job",
    document_path="complex_document.pdf",
    params={
        "mode": "advanced",
        "profile": "quality",
        "max_pages": 20,
        "extract_tables": True,
        "classify_document": True,
        "extract_metadata": True,
        "confidence_threshold": 0.9,
        "export_format": "json"
    }
)

# Access detailed results
if result.get("result_path"):
    import json
    with open(result["result_path"], 'r') as f:
        detailed_data = json.load(f)

    for page in detailed_data.get("pages", []):
        print(f"Page {page['page_num']}: {len(page['tokens'])} tokens")
        for token in page["tokens"][:5]:  # First 5 tokens
            print(f"  '{token['text']}' (confidence: {token['confidence']:.2%})")
```

## 🏗️ System Architecture

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
