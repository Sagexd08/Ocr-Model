# CurioScan Documentation

Welcome to the CurioScan documentation. CurioScan is a production-grade OCR system for ingesting, understanding, and exporting documents with high precision and accuracy.

## Overview

CurioScan combines advanced OCR technologies, machine learning, and intelligent document processing to provide:

- Intelligent document classification and understanding
- Hybrid OCR for both digital and scanned documents
- Advanced table detection and reconstruction
- High-precision text extraction and normalization
- Human-in-the-loop review and model retraining
- Scalable, containerized deployment options

## Documentation Sections

### User Guides
- [Installation and Deployment Guide](installation.md) - How to install, configure, and deploy CurioScan
- [API Reference](api_reference.md) - Complete API documentation
- [Developer Guide](developer_guide.md) - Guide for developers working with CurioScan

### Technical Documentation
- [System Architecture](architecture.md) - Overview of CurioScan's architecture
- [Pipeline Processors](pipeline_processors.md) - Detailed documentation of processing pipeline components

## Quick Start Guide

1. Build and run with Docker Compose:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

2. Open the interfaces:
   - API Documentation: http://localhost:8000/docs
   - Streamlit Demo: http://localhost:8501

3. Upload a document for processing via:
   - Streamlit interface
   - API endpoint at `POST /upload`
   - Command line: `curl -F "file=@/path/to/document.pdf" http://localhost:8000/upload`

4. Retrieve results via:
   - Streamlit interface
   - API endpoint at `GET /result/{job_id}`

## API Overview

CurioScan provides a RESTful API for document processing:

- `POST /upload` - Upload a document for processing
- `GET /status/{job_id}` - Check processing status
- `GET /result/{job_id}` - Get processing results
- `GET /review/{job_id}` - Get items that need review
- `POST /review/{job_id}` - Submit review corrections

Detailed API documentation is available at the [API Reference](api_reference.md) or via the `/docs` endpoint when the API is running.

## Developer Resources

- [Developer Guide](developer_guide.md) - Comprehensive guide for development
- [Pipeline Processors](pipeline_processors.md) - Documentation for pipeline components
- [Architecture Overview](architecture.md) - System architecture documentation

## Commands Reference

Common commands for working with CurioScan:

- **Docker Workflow**:
  - `make docker-build` - Build all Docker images
  - `make docker-up` - Start all services
  - `make docker-down` - Stop all services

- **Development**:
  - `make install-dev` - Install development dependencies
  - `make test` - Run all tests
  - `make lint` - Run linting checks
  - `make docs` - Build documentation

For a complete list of commands, see the [Makefile](../Makefile) or run `make help`.

## Modules
Autodoc sections can be expanded later to include API and worker modules.