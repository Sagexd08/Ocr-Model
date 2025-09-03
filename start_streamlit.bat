@echo off
echo Starting OCR System Streamlit Interface...
echo.
echo Open your browser and go to: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
python -m streamlit run streamlit_app.py --server.port 8501 --server.headless false
pause
