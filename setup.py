#!/usr/bin/env python
"""
CurioScan OCR Project Setup Script

This script sets up the CurioScan OCR project for running without Docker.
It performs the following tasks:
1. Creates necessary directories
2. Sets up environment variables
3. Installs dependencies
4. Configures database if needed
5. Provides options for running API and Streamlit separately or together
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor

# Paths
CURRENT_DIR = Path(os.getcwd())
PROJECT_ROOT = CURRENT_DIR
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_demo"
API_DIR = PROJECT_ROOT / "api"
WORKER_DIR = PROJECT_ROOT / "worker"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {text} ==={Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def run_command(command, cwd=None, env=None):
    """Run a command and return result"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, you have {sys.version}")
        return False
    
    print_success(f"Python version {sys.version} OK")
    return True

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    directories = [
        MODELS_DIR,
        DATA_DIR / "storage" / "input",
        DATA_DIR / "storage" / "output",
        DATA_DIR / "storage" / "temp",
        STREAMLIT_DIR / ".streamlit" / "cache",
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print_success(f"Created {directory}")
        except Exception as e:
            print_error(f"Failed to create {directory}: {str(e)}")

def setup_environment():
    """Setup environment variables"""
    print_header("Setting Up Environment")
    
    # Create .env file if it doesn't exist
    if not ENV_FILE.exists():
        with open(ENV_FILE, "w") as f:
            f.write("""# CurioScan OCR Environment Variables

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
ENVIRONMENT=production

# Database Configuration
DATABASE_URL=sqlite:///./curioscan_test.db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Storage Configuration
STORAGE_PATH=./data/storage
INPUT_PATH=./data/storage/input
OUTPUT_PATH=./data/storage/output
TEMP_PATH=./data/storage/temp

# Model Configuration
MODELS_PATH=./models

# Processing Configuration
MAX_WORKERS=4
CONFIDENCE_THRESHOLD=0.8
ENABLE_TABLE_DETECTION=true
ENABLE_FORM_DETECTION=true

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Streamlit Configuration
STREAMLIT_PORT=8501
CURIOSCAN_API_URL=http://localhost:8000
""")
        print_success("Created .env file")
    else:
        print_info(".env file already exists")

    # Create .streamlit/config.toml if it doesn't exist
    streamlit_config = STREAMLIT_DIR / ".streamlit" / "config.toml"
    if not streamlit_config.exists():
        (STREAMLIT_DIR / ".streamlit").mkdir(parents=True, exist_ok=True)
        with open(streamlit_config, "w") as f:
            f.write("""[server]
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = 8501

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[logger]
level = "info"
messageFormat = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""")
        print_success("Created Streamlit config")
    else:
        print_info("Streamlit config already exists")

def install_dependencies(args):
    """Install dependencies"""
    print_header("Installing Dependencies")
    
    # Install API dependencies
    if args.all or args.api:
        print_info("Installing API dependencies...")
        success, output = run_command("pip install -r requirements.txt", cwd=PROJECT_ROOT)
        if success:
            print_success("API dependencies installed")
        else:
            print_error(f"Failed to install API dependencies: {output}")
    
    # Install Streamlit dependencies
    if args.all or args.streamlit:
        print_info("Installing Streamlit dependencies...")
        success, output = run_command("pip install -r requirements.txt", cwd=STREAMLIT_DIR)
        if success:
            print_success("Streamlit dependencies installed")
        else:
            print_error(f"Failed to install Streamlit dependencies: {output}")

def setup_database():
    """Setup and migrate database"""
    print_header("Setting Up Database")
    
    # Run database migrations
    print_info("Running database migrations...")
    success, output = run_command("alembic upgrade head", cwd=PROJECT_ROOT)
    if success:
        print_success("Database migrations completed")
    else:
        print_error(f"Failed to run database migrations: {output}")

def run_api_server(debug=False):
    """Run the API server"""
    env = os.environ.copy()
    env["DEBUG"] = "true" if debug else "false"
    
    cmd = "uvicorn api.main:app --reload --host 0.0.0.0 --port 8000" if debug else \
          "uvicorn api.main:app --host 0.0.0.0 --port 8000"
    
    print_info(f"Starting API server on http://localhost:8000 (Debug: {debug})")
    
    process = subprocess.Popen(
        cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    # Wait a bit for server to start
    time.sleep(3)
    
    # Check if process is still running
    if process.poll() is None:
        print_success("API server running")
        return process
    else:
        output, _ = process.communicate()
        print_error(f"API server failed to start: {output}")
        return None

def run_celery_worker():
    """Run Celery worker"""
    print_info("Starting Celery worker...")
    
    process = subprocess.Popen(
        "celery -A worker.celery_app worker --loglevel=info",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    # Wait a bit for worker to start
    time.sleep(3)
    
    # Check if process is still running
    if process.poll() is None:
        print_success("Celery worker running")
        return process
    else:
        output, _ = process.communicate()
        print_error(f"Celery worker failed to start: {output}")
        return None

def run_streamlit_app():
    """Run Streamlit app"""
    print_info("Starting Streamlit app on http://localhost:8501")
    
    process = subprocess.Popen(
        "streamlit run app.py",
        cwd=STREAMLIT_DIR,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait a bit for app to start
    time.sleep(5)
    
    # Check if process is still running
    if process.poll() is None:
        print_success("Streamlit app running")
        # Open browser
        webbrowser.open("http://localhost:8501")
        return process
    else:
        output, _ = process.communicate()
        print_error(f"Streamlit app failed to start: {output}")
        return None

def run_full_stack(debug=False):
    """Run the full stack"""
    print_header("Starting CurioScan OCR Stack")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Start Redis if needed
        redis_running = False
        try:
            subprocess.run(["redis-cli", "ping"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            redis_running = True
            print_success("Redis server already running")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_warning("Redis server not running. Starting Redis...")
            redis_process = subprocess.Popen(
                "redis-server", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            time.sleep(2)
            redis_running = redis_process.poll() is None
            if redis_running:
                print_success("Redis server started")
            else:
                print_error("Failed to start Redis server. Make sure Redis is installed.")
                return
        
        # Start API server, Celery worker, and Streamlit app
        api_future = executor.submit(run_api_server, debug)
        celery_future = executor.submit(run_celery_worker)
        # Give API server time to start before launching Streamlit
        time.sleep(3)  
        streamlit_future = executor.submit(run_streamlit_app)
        
        # Wait for futures to complete
        api_process = api_future.result()
        celery_process = celery_future.result()
        streamlit_process = streamlit_future.result()
        
        if api_process and celery_process and streamlit_process:
            print_success("CurioScan OCR stack is running!")
            print_info("API server:    http://localhost:8000")
            print_info("Streamlit app: http://localhost:8501")
            
            try:
                print_info("Press Ctrl+C to stop...")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print_info("\nStopping all processes...")
                for process in [api_process, celery_process, streamlit_process]:
                    if process:
                        process.terminate()
                print_success("All processes stopped")
        else:
            print_error("Failed to start the complete stack")

def main():
    parser = argparse.ArgumentParser(description="CurioScan OCR Setup Script")
    
    # Setup options
    parser.add_argument("--setup", action="store_true", help="Setup directories and environment")
    parser.add_argument("--deps", action="store_true", help="Install dependencies")
    parser.add_argument("--db", action="store_true", help="Setup database")
    
    # Run options
    parser.add_argument("--api", action="store_true", help="Run API server only")
    parser.add_argument("--worker", action="store_true", help="Run Celery worker only")
    parser.add_argument("--streamlit", action="store_true", help="Run Streamlit app only")
    parser.add_argument("--all", action="store_true", help="Run full stack")
    
    # Additional options
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # If no specific options, default to full setup and run
    if not any(vars(args).values()):
        args.setup = True
        args.deps = True
        args.db = True
        args.all = True
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup
    if args.setup:
        create_directories()
        setup_environment()
    
    # Install dependencies
    if args.deps:
        install_dependencies(args)
    
    # Setup database
    if args.db:
        setup_database()
    
    # Run components
    if args.all:
        run_full_stack(debug=args.debug)
    else:
        if args.api:
            api_process = run_api_server(debug=args.debug)
            if api_process:
                try:
                    api_process.wait()
                except KeyboardInterrupt:
                    api_process.terminate()
                    print_info("API server stopped")
        
        if args.worker:
            worker_process = run_celery_worker()
            if worker_process:
                try:
                    worker_process.wait()
                except KeyboardInterrupt:
                    worker_process.terminate()
                    print_info("Celery worker stopped")
        
        if args.streamlit:
            streamlit_process = run_streamlit_app()
            if streamlit_process:
                try:
                    streamlit_process.wait()
                except KeyboardInterrupt:
                    streamlit_process.terminate()
                    print_info("Streamlit app stopped")

if __name__ == "__main__":
    main()
