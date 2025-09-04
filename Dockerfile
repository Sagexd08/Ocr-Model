# Enterprise OCR Processing System - Multi-stage Production Dockerfile
FROM python:3.9-slim as base

# Set metadata
LABEL maintainer="OCR System Team"
LABEL version="2.0.0"
LABEL description="Enterprise OCR Processing System with Advanced Dark Mode Streamlit Interface"

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV STREAMLIT_SERVER_PORT=8505
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_THEME_BASE=dark
ENV STREAMLIT_THEME_PRIMARY_COLOR="#00D4FF"
ENV STREAMLIT_THEME_BACKGROUND_COLOR="#0E1117"
ENV STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#1E2329"
ENV STREAMLIT_THEME_TEXT_COLOR="#FAFAFA"

# Install system dependencies for OCR and image processing
RUN apt-get update && apt-get install -y \
    # OpenCV and image processing dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libjpeg62-turbo-dev \
    libpng16-16 \
    # PDF processing
    poppler-utils \
    libpoppler-cpp-dev \
    libpoppler-dev \
    # Font libraries
    fontconfig \
    fonts-dejavu-core \
    fonts-liberation \
    # Network and utility tools
    wget \
    curl \
    unzip \
    # Build tools (needed for some Python packages)
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    # Additional libraries for PaddleOCR
    libffi-dev \
    libssl-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create application user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 ocr_user

# Set working directory
WORKDIR /app

# Copy requirements files first for better Docker layer caching
COPY requirements.txt requirements_streamlit.txt ./

# Install Python dependencies in stages for better caching
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies first
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    pillow>=10.0.0 \
    opencv-python>=4.8.0 \
    streamlit>=1.28.0 \
    plotly>=5.15.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.22.0

# Install OCR dependencies
RUN pip install --no-cache-dir \
    paddlepaddle>=2.5.0 \
    paddleocr>=2.7.0 \
    PyMuPDF>=1.23.0 \
    pdf2image>=1.16.0

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_streamlit.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p \
    data/storage/input \
    data/storage/output \
    data/storage/cache \
    logs \
    output \
    temp \
    .streamlit \
    .paddlex \
    && chown -R ocr_user:ocr_user /app \
    && chmod -R 755 /app \
    && chmod -R 777 /app/data \
    && chmod -R 777 /app/output \
    && chmod -R 777 /app/logs \
    && chmod -R 777 /app/temp

# Copy Streamlit configuration
COPY .streamlit/config.toml .streamlit/config.toml

# Pre-download PaddleOCR models (optional, for faster startup)
RUN python -c "import paddleocr; paddleocr.PaddleOCR(use_angle_cls=True, lang='en')" || echo "Model download will happen at runtime"

# Switch to non-root user
USER ocr_user

# Expose ports
EXPOSE 8505 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8505/_stcore/health || exit 1

# Default command - run the advanced OCR Streamlit app
CMD ["streamlit", "run", "advanced_ocr_app.py", "--server.port=8505", "--server.address=0.0.0.0", "--server.headless=true"]
