# CurioScan OCR - Streamlit Demo

A production-ready Streamlit application for interacting with the CurioScan OCR intelligent document processing system.

## Features

- **Document Upload**: Upload documents in various formats (PDF, images, Word, Excel)
- **Processing Options**: Configure OCR settings, extraction options, and output formats
- **Results Visualization**: View extracted text, tables, and form fields with visualization tools
- **Data Export**: Download processed results in multiple formats (JSON, CSV, Excel, PDF)
- **Job History**: Track and revisit previously processed documents
- **Processing Analytics**: View statistics on document processing efficiency and accuracy

## Architecture

The application follows a modern production-ready architecture:

- **Multi-layered UI**: Streamlit components for different functionalities
- **API Client**: Dedicated client for communicating with the OCR backend API
- **Visualization Components**: Specialized visualization tools for different data types
- **Analytics Dashboard**: Real-time processing statistics
- **Error Handling**: Comprehensive error handling and user feedback
- **Security Features**: Input validation, CORS protection, and secure communication

## Setup

### Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
export CURIOSCAN_API_URL=http://localhost:8000  # URL of the OCR API
export MAX_FILE_SIZE_MB=50                      # Maximum file size for upload
export DEMO_MODE=false                          # Enable/disable demo mode
```

3. Run the application:

```bash
streamlit run app.py
```

### Docker Deployment

1. Build the Docker image:

```bash
docker build -t curioscan-streamlit:latest .
```

2. Run the container:

```bash
docker run -p 8501:8501 -e CURIOSCAN_API_URL=http://api:8000 curioscan-streamlit:latest
```

### Configuration

Configuration can be set through environment variables or by editing `.streamlit/config.toml`:

- `CURIOSCAN_API_URL`: URL of the CurioScan OCR API
- `MAX_FILE_SIZE_MB`: Maximum file size for document upload
- `DEMO_MODE`: Enable demo mode with sample data
- `ENABLE_ANALYTICS`: Enable/disable analytics tracking
- `CACHE_DIR`: Directory for caching processed results

## Usage

1. **Upload Document**: Use the sidebar to upload a document
2. **Configure Processing**: Set OCR mode, confidence threshold, and extraction options
3. **View Results**: Once processing is complete, explore the extracted data through the tabs
4. **Download Results**: Export the processed data in your preferred format
5. **Review History**: Access previously processed documents from the sidebar

## Production Deployment

For production deployment:

1. Use the provided Dockerfile which includes:
   - Multi-stage build for smaller image size
   - Non-root user for security
   - Health checks
   - Optimized dependency installation

2. Set appropriate environment variables for your production environment

3. Configure a reverse proxy (like Nginx) for SSL termination and additional security

4. Set up monitoring and logging solutions

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

Proprietary - All Rights Reserved © 2025 CurioScan
