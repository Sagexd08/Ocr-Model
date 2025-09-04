# 🐳 Docker Deployment Guide - Enterprise OCR Processing System

## 🚀 Quick Start

### One-Command Deployment
```bash
python deploy-docker.py
```

### Manual Deployment
```bash
# Build and start all services
docker-compose up --build -d

# Access the application
# Streamlit UI: http://localhost:8505
# API Documentation: http://localhost:8001/docs
```

## 📋 Prerequisites

### System Requirements
- **Docker**: Version 20.10+ 
- **Docker Compose**: Version 2.0+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 4 cores minimum (8 cores recommended)

### Supported Platforms
- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- **Windows**: Windows 10/11 with WSL2 and Docker Desktop
- **macOS**: macOS 11+ (Intel and Apple Silicon)
- **Cloud**: AWS, Azure, GCP, DigitalOcean

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  Streamlit App  │    │   FastAPI       │
│   Port: 80      │────│   Port: 8505    │────│   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │  PostgreSQL     │
                       │   Port: 6379    │    │   Port: 5432    │
                       └─────────────────┘    └─────────────────┘
```

## 🔧 Deployment Options

### 1. Production Deployment
```bash
# Full production stack with all services
docker-compose up --build -d

# Services included:
# - Enterprise OCR Streamlit App (8505)
# - FastAPI Backend (8001)
# - Redis Cache (6379)
# - PostgreSQL Database (5432)
# - Nginx Reverse Proxy (80)
```

### 2. Development Deployment
```bash
# Simplified development environment
docker-compose -f docker-compose.dev.yml up --build -d

# Services included:
# - OCR Streamlit App (8505)
# - Redis Cache (6379)
```

### 3. Streamlit Only
```bash
# Just the OCR application
docker build -t enterprise-ocr .
docker run -p 8505:8505 enterprise-ocr
```

## 🧪 Testing & Validation

### Automated Testing
```bash
# Run comprehensive test suite
python docker-test.py

# Test report saved to: docker-test-report.json
```

### Manual Testing
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Test endpoints
curl http://localhost:8505  # Streamlit UI
curl http://localhost:8001/docs  # API Documentation
curl http://localhost:80  # Nginx Proxy
```

## 📊 Service Details

### 🎨 Streamlit App (Port 8505)
- **Advanced Dark Mode UI** with professional styling
- **Real-time OCR Processing** with PaddleOCR
- **Interactive Analytics** with Plotly visualizations
- **Multiple Export Formats** (JSON, CSV, TXT, Markdown)

### 🔌 FastAPI Backend (Port 8001)
- **RESTful API** for programmatic access
- **Interactive Documentation** at `/docs`
- **Health Checks** and monitoring endpoints
- **Async Processing** for better performance

### 🗄️ Redis Cache (Port 6379)
- **Session Management** for user state
- **Job Queuing** for background processing
- **Performance Caching** for faster responses
- **Memory Optimization** with LRU eviction

### 🗃️ PostgreSQL Database (Port 5432)
- **Job History** and analytics persistence
- **User Management** and authentication
- **Performance Metrics** storage
- **Backup and Recovery** capabilities

### 🌐 Nginx Proxy (Port 80)
- **Load Balancing** for production traffic
- **SSL Termination** for HTTPS support
- **Static File Serving** for optimized delivery
- **Request Routing** and path-based routing

## 🔧 Configuration

### Environment Variables
```bash
# Core Configuration
PYTHONPATH=/app
PYTHONUNBUFFERED=1
OCR_PROFILE=balanced  # performance, quality, balanced
LOG_LEVEL=INFO

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8505
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_THEME_BASE=dark
STREAMLIT_THEME_PRIMARY_COLOR=#00D4FF

# Database Configuration
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://ocr_user:ocr_password@postgres:5432/ocr_system
```

### Volume Mounts
```yaml
volumes:
  - ./data:/app/data:rw          # Input/output data
  - ./output:/app/output:rw      # Processed results
  - ./logs:/app/logs:rw          # Application logs
  - ./temp:/app/temp:rw          # Temporary files
```

## 🛠️ Management Commands

### Start Services
```bash
docker-compose up -d                    # Start all services
docker-compose up -d ocr-app           # Start specific service
docker-compose -f docker-compose.dev.yml up -d  # Development mode
```

### Stop Services
```bash
docker-compose down                     # Stop all services
docker-compose stop ocr-app            # Stop specific service
docker-compose down -v                 # Stop and remove volumes
```

### View Logs
```bash
docker-compose logs -f                  # All services
docker-compose logs -f ocr-app         # Specific service
docker-compose logs --tail=100 ocr-app # Last 100 lines
```

### Scale Services
```bash
docker-compose up -d --scale ocr-app=3  # Scale to 3 instances
docker-compose up -d --scale api=2      # Scale API backend
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
netstat -tulpn | grep :8505

# Kill the process or change port in docker-compose.yml
```

#### 2. Build Failures
```bash
# Clean build cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

#### 3. Permission Issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER ./data ./output ./logs

# Or use Docker with proper user mapping
```

#### 4. Memory Issues
```bash
# Increase Docker memory limit in Docker Desktop
# Or add memory limits to docker-compose.yml
```

### Health Checks
```bash
# Check service health
docker-compose ps

# Manual health check
curl -f http://localhost:8505/_stcore/health
curl -f http://localhost:8001/health
```

## 📈 Performance Tuning

### Resource Limits
```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
    reservations:
      memory: 4G
      cpus: '2.0'
```

### Optimization Tips
- **Use SSD storage** for better I/O performance
- **Allocate sufficient RAM** (16GB+ recommended)
- **Enable Docker BuildKit** for faster builds
- **Use multi-stage builds** for smaller images
- **Configure proper logging** to prevent disk space issues

## 🔐 Security Considerations

### Production Security
- **Change default passwords** in docker-compose.yml
- **Use environment files** for sensitive data
- **Enable SSL/TLS** with proper certificates
- **Configure firewall rules** for exposed ports
- **Regular security updates** for base images

### Network Security
```yaml
networks:
  ocr-network:
    driver: bridge
    internal: true  # Isolate from external networks
```

## 📚 Additional Resources

- **GitHub Repository**: https://github.com/Sagexd08/Ocr-Model
- **Docker Hub**: (Coming soon)
- **Documentation**: README.md
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

<div align="center">

**🐳 Enterprise OCR Processing System - Docker Edition**

*Professional Document Processing in Containers*

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Compose](https://img.shields.io/badge/Compose-v2.0+-green.svg)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Built with ❤️ for the containerized world*

</div>
