@echo off
setlocal
REM Set Redis broker/backend then start worker
set CELERY_BROKER_URL=redis://localhost:6379/0
set CELERY_RESULT_BACKEND=redis://localhost:6379/1
celery -A api.celery_app.celery_app worker -Q default -l INFO --pool=solo
endlocal

