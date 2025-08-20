# CurioScan - Production-Grade OCR System

CurioScan is a comprehensive OCR system designed for production use, featuring intelligent document classification, hybrid OCR processing, table extraction, and human-in-the-loop review capabilities.

## 🚀 Quick Start

### Local Development with Docker Compose
```bash
# Clone and setup
git clone <repository-url>
cd OCR(Freelance)

# Start all services (API, Worker, Redis, MinIO, Streamlit)
docker-compose up

# Access Streamlit demo at http://localhost:8501
# API documentation at http://localhost:8000/docs
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis and MinIO (or use docker-compose for just these)
docker-compose up redis minio

# Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start worker
celery -A worker.celery_app worker --loglevel=info

# Start Streamlit demo
streamlit run streamlit_demo/app.py
```

### Sample Processing
```bash
# Run end-to-end demo
./demo_process.sh

# Train model
python training/train.py --config configs/demo.yaml

# Evaluate model
python evaluation/evaluate.py --dataset demo_holdout
```

## 📁 Repository Structure

```
OCR(Freelance)/
├── training/           # Training scripts and configs
├── models/            # Model definitions and checkpoints
├── api/               # FastAPI backend
├── worker/            # Celery worker pipeline
├── streamlit_demo/    # Streamlit demo application
├── deploy/            # Deployment configurations
├── tests/             # Test suite
├── evaluation/        # Evaluation scripts
├── configs/           # Configuration files
├── data/              # Sample data and datasets
├── docs/              # Documentation
├── requirements.txt   # Python dependencies
├── docker-compose.yml # Local development setup
├── demo_process.sh    # End-to-end demo script
└── README.md          # This file
```

## 🎯 Key Features

### Document Processing
- **Multi-format support**: PDFs (digital + scanned), DOCX, TIFF/JPEG/PNG, photographed pages
- **Intelligent classification**: Automatic render type detection (digital_pdf, scanned_image, photograph, docx, form, table_heavy, invoice, handwritten)
- **Hybrid OCR**: Native text extraction for digital PDFs, advanced OCR for scanned content
- **Table extraction**: Robust table detection and reconstruction with rowspan/colspan handling
- **Provenance tracking**: Complete lineage from source to output with bounding boxes and confidence scores

### Human-in-the-Loop Review
- **Confidence-based flagging**: Automatic identification of low-confidence regions
- **Interactive review UI**: Edit, accept, or reject OCR results
- **Retraining pipeline**: Feed corrections back into model training

### Production-Ready Architecture
- **Scalable workers**: Redis + Celery for async processing
- **API-first design**: RESTful API with webhooks and status tracking
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Deployment**: Docker, Kubernetes Helm charts, auto-scaling

## 📊 Output Schema

All exports follow this exact schema with complete provenance:

```json
{
  "rows": [
    {
      "row_id": "string",
      "page": 1,
      "region_id": "string", 
      "bbox": [x1, y1, x2, y2],
      "columns": {"col_name": "value"},
      "provenance": {
        "file": "string",
        "page": 1,
        "bbox": [x1, y1, x2, y2],
        "token_ids": [1, 2, 3],
        "confidence": 0.95
      },
      "needs_review": false
    }
  ]
}
```

## 🔧 API Endpoints

- `POST /upload` → `{ job_id }`
- `GET /status/{job_id}` → `{ status, progress, preview }`
- `GET /result/{job_id}` → downloadable CSV/XLSX/JSON + provenance
- `POST /webhooks/register` → register callback
- `GET /review/{job_id}` → paginated review payload for UI
- `POST /retrain-trigger` → trigger retraining with accepted corrections

## 🎨 Streamlit Demo Features

- **Pipeline visualization**: Live status and timing for each processing stage
- **OCR overlay viewer**: Toggle bounding boxes, tokens, confidence heatmaps
- **Interactive table editor**: Edit cells, merge/split, add/remove rows
- **Provenance inspector**: Click any cell to see source location and OCR details
- **Training dashboard**: Loss curves, F1 scores, evaluation reports
- **Export options**: CSV, XLSX, JSON, complete provenance ZIP

## 🏋️ Training & Evaluation

### Performance Targets
- Digital PDF token F1 ≥ 0.99
- Scanned pages token F1 ≥ 0.98  
- Table cell reconstruction ≥ 0.95
- Single-page inference < 2s on GPU

### Training Features
- **Multi-model support**: LayoutLMv3, TrOCR, DocTR, Tesseract fallback
- **Advanced techniques**: Ensembling, knowledge distillation, active learning
- **Robust augmentation**: Perspective warp, blur, artifacts, occlusions
- **Distributed training**: DDP, AMP, checkpointing, SLURM/K8s job examples
- **Hyperparameter optimization**: Optuna integration
- **Model export**: ONNX, TorchScript, quantization examples

## 🔒 Security & Privacy

- **TLS endpoints**: Secure API communication
- **Encryption**: S3/MinIO storage encryption
- **On-premise option**: Complete air-gapped deployment
- **PII detection**: Optional redaction capabilities
- **Access controls**: Token-based or OAuth integration

## 🚀 Deployment Options

### Local Development
```bash
docker-compose up
```

### Production Kubernetes
```bash
helm install curioscan deploy/helm/curioscan
```

### Monitoring
- Prometheus metrics collection
- Grafana dashboards for performance monitoring
- Health checks and alerting

## 🧠 LLM Integration (Optional)

LLMs are used strictly for structural disambiguation and normalization - never to invent missing data:

**System Prompt**: "You are a strict normalizer. Input: OCR tokens with bbox and confidence. Output: JSON matching the schema. Do not invent missing values. If unsure, set needs_review=true. Output JSON only."

## 📈 Evaluation Metrics

- Token-level precision/recall/F1
- Region detection mAP (IoU thresholds)
- Table reconstruction accuracy (cell-level)
- End-to-end field exact-match accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- Documentation: `/docs`
- Issues: GitHub Issues
- API Reference: `http://localhost:8000/docs` when running

---

**Built with**: PyTorch, HuggingFace Transformers, FastAPI, Celery, Streamlit, OpenCV, pdfplumber
