@echo off
setlocal
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
endlocal

