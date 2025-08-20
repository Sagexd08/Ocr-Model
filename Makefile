# CurioScan Makefile
# Development and deployment automation

.PHONY: help install install-dev test test-unit test-integration test-e2e lint format clean build docker-build docker-up docker-down deploy docs

# Default target
help:
	@echo "CurioScan Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-e2e         Run end-to-end tests"
	@echo "  test-performance Run performance tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security-check   Run security checks"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build all Docker images"
	@echo "  docker-up        Start all services with Docker Compose"
	@echo "  docker-down      Stop all services"
	@echo "  docker-logs      Show logs from all services"
	@echo ""
	@echo "Development:"
	@echo "  run-api          Run API server locally"
	@echo "  run-worker       Run Celery worker locally"
	@echo "  run-demo         Run Streamlit demo locally"
	@echo ""
	@echo "Training:"
	@echo "  train            Run model training"
	@echo "  evaluate         Run model evaluation"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-staging   Deploy to staging environment"
	@echo "  deploy-prod      Deploy to production environment"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean up temporary files"
	@echo "  docs             Build documentation"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "not slow"

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

test-performance:
	pytest tests/performance/ -v -m performance --benchmark-only

test-coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term

# Code Quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .
	isort .

format-check:
	black --check .
	isort --check-only .

type-check:
	mypy --ignore-missing-imports .

security-check:
	bandit -r . -f json
	safety check

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f

# Development servers
run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-worker:
	celery -A worker.celery_app worker --loglevel=info --concurrency=2

run-flower:
	celery -A worker.celery_app flower --port=5555

run-demo:
	streamlit run streamlit_demo/app.py --server.port 8501

# Training and evaluation
train:
	python training/train.py --config configs/renderer_classifier.yaml

train-ocr:
	python training/train.py --config configs/ocr_model.yaml

evaluate:
	python evaluation/evaluate.py --config configs/evaluation.yaml --model-path models/best_model.pth

# Database operations
db-migrate:
	alembic upgrade head

db-reset:
	alembic downgrade base
	alembic upgrade head

db-seed:
	python scripts/seed_database.py

# Data operations
download-models:
	python scripts/download_models.py

prepare-data:
	python scripts/prepare_training_data.py

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Deployment
deploy-staging:
	@echo "Deploying to staging..."
	docker-compose -f docker-compose.staging.yml up -d

deploy-prod:
	@echo "Deploying to production..."
	docker-compose -f docker-compose.prod.yml up -d

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf build/

clean-docker:
	docker system prune -f
	docker volume prune -f

# CI/CD helpers
ci-install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

ci-test:
	pytest tests/ --cov=. --cov-report=xml --cov-report=term -v

ci-lint:
	black --check .
	isort --check-only .
	flake8 .
	mypy --ignore-missing-imports .

ci-security:
	bandit -r . -f json
	safety check

# Release
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major

# Monitoring
logs-api:
	docker-compose logs -f api

logs-worker:
	docker-compose logs -f worker

logs-all:
	docker-compose logs -f

# Health checks
health-check:
	curl -f http://localhost:8000/health || exit 1

status:
	@echo "Service Status:"
	@echo "==============="
	@curl -s http://localhost:8000/health | jq . || echo "API: DOWN"
	@curl -s http://localhost:5555/api/workers | jq . || echo "Workers: DOWN"
	@curl -s http://localhost:9000/minio/health/live || echo "MinIO: DOWN"
