@echo off
setlocal
REM Use SQLite/SQLAlchemy transport (default in api/celery_app.py)
celery -A api.celery_app.celery_app worker -Q default -l INFO --pool=solo
endlocal

