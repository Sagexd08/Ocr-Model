#!/usr/bin/env bash
set -euo pipefail
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/1
celery -A api.celery_app.celery_app worker -Q default -l INFO

