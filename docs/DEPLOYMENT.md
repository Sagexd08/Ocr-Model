# 🚀 Deployment Guide

## Overview

This guide covers various deployment options for the Enterprise OCR Processing System, from local development to production cloud deployments.

## 🔧 Local Development

### Prerequisites
- Python 3.8+
- 8GB RAM (16GB recommended)
- 10GB free disk space
- Git

### Quick Setup

```bash
# Clone repository
git clone https://github.com/Sagexd08/Ocr-Model.git
cd Ocr-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_streamlit.txt

# Launch application
python launch_advanced_ocr.py
```

## 🐳 Docker Deployment

### Single Container

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8505 8000

# Start application
CMD ["python", "launch_advanced_ocr.py"]
```

```bash
# Build and run
docker build -t enterprise-ocr .
docker run -p 8505:8505 -p 8000:8000 enterprise-ocr
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ocr-app:
    build: .
    ports:
      - "8505:8505"
      - "8000:8000"
    environment:
      - OCR_PROFILE=balanced
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ocr_system
      POSTGRES_USER: ocr_user
      POSTGRES_PASSWORD: ocr_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ocr-app

volumes:
  redis_data:
  postgres_data:
```

```bash
# Deploy with Docker Compose
docker-compose up -d --build

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale ocr-app=3
```

## ☸️ Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ocr-system

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ocr-config
  namespace: ocr-system
data:
  OCR_PROFILE: "balanced"
  LOG_LEVEL: "INFO"
  MAX_UPLOAD_SIZE: "200"
  CACHE_ENABLED: "true"
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-app
  namespace: ocr-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ocr-app
  template:
    metadata:
      labels:
        app: ocr-app
    spec:
      containers:
      - name: ocr-app
        image: enterprise-ocr:latest
        ports:
        - containerPort: 8505
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: ocr-config
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: output-volume
          mountPath: /app/output
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: ocr-data-pvc
      - name: output-volume
        persistentVolumeClaim:
          claimName: ocr-output-pvc
```

### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ocr-service
  namespace: ocr-system
spec:
  selector:
    app: ocr-app
  ports:
  - name: web
    port: 8505
    targetPort: 8505
  - name: api
    port: 8000
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ocr-ingress
  namespace: ocr-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - ocr.yourdomain.com
    secretName: ocr-tls
  rules:
  - host: ocr.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ocr-service
            port:
              number: 8505
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: ocr-service
            port:
              number: 8000
```

### Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n ocr-system
kubectl get services -n ocr-system
kubectl get ingress -n ocr-system

# View logs
kubectl logs -f deployment/ocr-app -n ocr-system

# Scale deployment
kubectl scale deployment ocr-app --replicas=5 -n ocr-system
```

## 🌩️ Cloud Deployments

### AWS ECS

```json
{
  "family": "enterprise-ocr",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ocr-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/enterprise-ocr:latest",
      "portMappings": [
        {
          "containerPort": 8505,
          "protocol": "tcp"
        },
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OCR_PROFILE",
          "value": "balanced"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/enterprise-ocr",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: enterprise-ocr
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "8Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containers:
      - image: gcr.io/your-project/enterprise-ocr:latest
        ports:
        - containerPort: 8505
        env:
        - name: OCR_PROFILE
          value: "balanced"
        - name: PORT
          value: "8505"
        resources:
          limits:
            memory: "8Gi"
            cpu: "2"
```

```bash
# Deploy to Cloud Run
gcloud run deploy enterprise-ocr \
  --image gcr.io/your-project/enterprise-ocr:latest \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 2 \
  --max-instances 10 \
  --port 8505
```

### Azure Container Instances

```yaml
# azure-container.yaml
apiVersion: 2019-12-01
location: eastus
name: enterprise-ocr
properties:
  containers:
  - name: ocr-app
    properties:
      image: your-registry.azurecr.io/enterprise-ocr:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 8
      ports:
      - port: 8505
        protocol: TCP
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: OCR_PROFILE
        value: balanced
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8505
    - protocol: tcp
      port: 8000
    dnsNameLabel: enterprise-ocr
```

## 🔒 Production Considerations

### Security

```yaml
# Security configuration
security:
  # Enable HTTPS
  ssl:
    enabled: true
    cert_file: /etc/ssl/certs/ocr.crt
    key_file: /etc/ssl/private/ocr.key
  
  # Authentication
  auth:
    enabled: true
    type: "oauth2"
    providers:
      - google
      - github
  
  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
```

### Monitoring

```yaml
# Prometheus monitoring
monitoring:
  prometheus:
    enabled: true
    port: 9090
    metrics_path: /metrics
  
  # Health checks
  health_checks:
    liveness_probe:
      path: /health
      interval: 30s
    readiness_probe:
      path: /ready
      interval: 10s
```

### Scaling

```yaml
# Auto-scaling configuration
autoscaling:
  enabled: true
  min_replicas: 2
  max_replicas: 20
  target_cpu_utilization: 70
  target_memory_utilization: 80
  
  # Custom metrics
  custom_metrics:
    - name: processing_queue_length
      target_value: 10
```

### Backup and Recovery

```bash
# Database backup
kubectl create cronjob postgres-backup \
  --image=postgres:15-alpine \
  --schedule="0 2 * * *" \
  -- pg_dump -h postgres-service -U ocr_user ocr_system > /backup/db-$(date +%Y%m%d).sql

# Volume snapshots
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: ocr-data-snapshot
spec:
  source:
    persistentVolumeClaimName: ocr-data-pvc
EOF
```

## 🔧 Environment Variables

### Production Environment

```bash
# Core settings
export OCR_PROFILE=balanced
export LOG_LEVEL=INFO
export MAX_UPLOAD_SIZE=200
export CACHE_ENABLED=true

# Database
export DATABASE_URL=postgresql://user:pass@host:5432/ocr_system
export REDIS_URL=redis://redis-host:6379

# Security
export SECRET_KEY=your-secret-key
export JWT_SECRET=your-jwt-secret
export CORS_ORIGINS=https://yourdomain.com

# Monitoring
export SENTRY_DSN=your-sentry-dsn
export PROMETHEUS_ENABLED=true

# Storage
export S3_BUCKET=your-ocr-bucket
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

## 📊 Performance Tuning

### Resource Allocation

```yaml
# Resource recommendations
resources:
  small_deployment:
    cpu: "1000m"
    memory: "4Gi"
    replicas: 2
  
  medium_deployment:
    cpu: "2000m"
    memory: "8Gi"
    replicas: 5
  
  large_deployment:
    cpu: "4000m"
    memory: "16Gi"
    replicas: 10
```

### Optimization Tips

1. **CPU Optimization**
   - Use performance profile for speed
   - Enable multi-threading
   - Optimize image preprocessing

2. **Memory Optimization**
   - Implement result caching
   - Use streaming for large files
   - Configure garbage collection

3. **Storage Optimization**
   - Use SSD storage
   - Implement data compression
   - Regular cleanup of temporary files

4. **Network Optimization**
   - Enable gzip compression
   - Use CDN for static assets
   - Implement connection pooling
