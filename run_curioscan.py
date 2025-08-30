"""
Script to run the CurioScan OCR application (API + Streamlit)
"""
import os
import sys
import subprocess
import time
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("curioscan-runner")

# Get the directory of this script
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def run_api_server():
    """Run the API server"""
    logger.info("Starting API server...")
    api_process = subprocess.Popen(
        [sys.executable, "run_api_server.py"],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )
    return api_process

def run_streamlit_app():
    """Run the Streamlit app"""
    logger.info("Starting Streamlit app...")
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_demo/app.py"],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )
    return streamlit_process

def log_output(process, name):
    """Log the output from a process"""
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            logger.info(f"[{name}] {line.rstrip()}")

def main():
    """Main function to run both services"""
    try:
        logger.info("Starting CurioScan OCR application...")
        
        # Create data directories if they don't exist
        storage_path = SCRIPT_DIR / "data" / "storage"
        input_path = storage_path / "input"
        output_path = storage_path / "output"
        temp_path = storage_path / "temp"
        
        for path in [input_path, output_path, temp_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Start API server
        api_process = run_api_server()
        logger.info("API server started with PID: %d", api_process.pid)
        
        # Wait for API server to be ready
        time.sleep(3)
        
        # Start Streamlit app
        streamlit_process = run_streamlit_app()
        logger.info("Streamlit app started with PID: %d", streamlit_process.pid)
        
        # Monitor and log output from both processes
        processes = [
            (api_process, "API"),
            (streamlit_process, "Streamlit")
        ]
        
        try:
            # Wait for both processes to complete or for user to interrupt
            while all(process.poll() is None for process, _ in processes):
                time.sleep(0.1)
                
            # Check if any process exited
            for process, name in processes:
                if process.poll() is not None:
                    logger.error(f"{name} process exited with code {process.returncode}")
                    break
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down...")
            
        finally:
            # Terminate processes
            for process, name in processes:
                if process.poll() is None:
                    logger.info(f"Terminating {name} process...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"{name} process didn't terminate, killing...")
                        process.kill()
            
    except Exception as e:
        logger.error(f"Error running CurioScan OCR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
