# Installation and Deployment Guide

This guide covers how to install, configure, and deploy CurioScan in various environments.

## Local Development Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Git
- Make (optional, for convenience commands)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sagexd08/Ocr-Model.git
cd Ocr-Model
```

### Step 2: Install Dependencies

For development with all tools:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

For production dependencies only:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

Create a `.env` file in the project root with required configuration:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Storage
STORAGE_TYPE=local
STORAGE_PATH=./data/storage
# For S3/MinIO
# STORAGE_TYPE=s3
# S3_ENDPOINT=http://localhost:9000
# S3_ACCESS_KEY=minioadmin
# S3_SECRET_KEY=minioadmin
# S3_REGION=us-east-1
# S3_BUCKET=curioscan

# Database
DATABASE_URL=sqlite:///curioscan.db

# Worker
BROKER_URL=redis://localhost:6379/0
RESULT_BACKEND=redis://localhost:6379/0
```

### Step 4: Run Development Services

```bash
# Start required services (Redis, MinIO)
docker-compose up -d redis minio

# Run API in development mode
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run worker in another terminal
celery -A worker.celery_app worker --loglevel=info
```

## Docker Deployment

### Using Docker Compose

The easiest way to run the full system:

```bash
# Build all services
docker-compose build

# Start the entire system
docker-compose up -d

# Check logs
docker-compose logs -f
```

To stop all services:

```bash
docker-compose down
```

### Individual Services

You can also build and run individual services:

```bash
# Build and run only the API
docker-compose build api
docker-compose up -d api

# Scale workers for more throughput
docker-compose up -d --scale worker=3
```

## Production Deployment

### System Requirements

Minimum recommended specifications:
- 4 CPU cores
- 16GB RAM
- 100GB storage
- GPU recommended for high-volume processing

### Kubernetes Deployment

CurioScan can be deployed to Kubernetes using the provided Helm charts:

```bash
# Add the CurioScan Helm repository
helm repo add curioscan https://charts.curioscan.example.com
helm repo update

# Install the CurioScan chart
helm install curioscan curioscan/curioscan \
  --namespace curioscan \
  --create-namespace \
  --values my-values.yaml
```

Example `my-values.yaml`:

```yaml
api:
  replicas: 2
  resources:
    requests:
      cpu: 1
      memory: 2Gi
    limits:
      cpu: 2
      memory: 4Gi

worker:
  replicas: 3
  resources:
    requests:
      cpu: 2
      memory: 4Gi
    limits:
      cpu: 4
      memory: 8Gi
  gpu:
    enabled: true
    type: nvidia-tesla-t4
    count: 1

storage:
  type: s3
  s3:
    endpoint: https://s3.amazonaws.com
    bucket: my-curioscan-bucket
    region: us-east-1
    # Credentials should be provided via Kubernetes secrets

database:
  type: postgresql
  host: postgres.database.svc.cluster.local
  port: 5432
  database: curioscan
  # Credentials should be provided via Kubernetes secrets
```

### Cloud Deployment

#### AWS Deployment

For AWS deployment, use the provided CloudFormation template:

```bash
aws cloudformation create-stack \
  --stack-name curioscan \
  --template-body file://deploy/aws/template.yaml \
  --parameters file://deploy/aws/parameters.json \
  --capabilities CAPABILITY_IAM
```

#### Azure Deployment

For Azure deployment, use the provided ARM template:

```bash
az group create --name curioscan --location eastus
az deployment group create \
  --resource-group curioscan \
  --template-file deploy/azure/template.json \
  --parameters deploy/azure/parameters.json
```

## Configuration Options

### API Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `API_HOST` | Host to bind the API server | `0.0.0.0` |
| `API_PORT` | Port for the API server | `8000` |
| `DEBUG` | Enable debug mode | `False` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CORS_ORIGINS` | Allowed CORS origins | `["*"]` |
| `MAX_UPLOAD_SIZE` | Maximum upload size in MB | `100` |

### Worker Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BROKER_URL` | Celery broker URL | `redis://localhost:6379/0` |
| `RESULT_BACKEND` | Celery result backend | `redis://localhost:6379/0` |
| `WORKER_CONCURRENCY` | Number of concurrent worker processes | `Number of CPUs` |
| `TASK_TIME_LIMIT` | Maximum task execution time (seconds) | `3600` |
| `USE_GPU` | Enable GPU acceleration | `False` |
| `GPU_DEVICE` | GPU device ID | `0` |

### Storage Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `STORAGE_TYPE` | Storage type (`local` or `s3`) | `local` |
| `STORAGE_PATH` | Local storage path | `./data/storage` |
| `S3_ENDPOINT` | S3/MinIO endpoint URL | `http://localhost:9000` |
| `S3_ACCESS_KEY` | S3/MinIO access key | `minioadmin` |
| `S3_SECRET_KEY` | S3/MinIO secret key | `minioadmin` |
| `S3_REGION` | S3/MinIO region | `us-east-1` |
| `S3_BUCKET` | S3/MinIO bucket name | `curioscan` |

### Database Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DATABASE_URL` | Database connection URL | `sqlite:///curioscan.db` |

## Health Monitoring

For production deployments, health endpoints are available:

- API health check: `GET /health`
- Worker health check: `GET /health/worker`
- Storage health check: `GET /health/storage`

Use these endpoints with your monitoring system to ensure system availability.
