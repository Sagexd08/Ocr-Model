# Running CurioScan OCR Without Docker

This guide explains how to run the CurioScan OCR project without Docker using the provided setup scripts.

## Prerequisites

- Python 3.8+ installed
- Redis server (for the Celery worker)
- pip package manager

## Setup Scripts

We provide several scripts to make it easier to run CurioScan OCR without Docker:

1. **setup.py** - The main Python setup script that handles all aspects of setting up and running the project
2. **start.bat** - A Windows batch script for easy command-line usage
3. **start.ps1** - A PowerShell script for Windows users who prefer PowerShell

## Quick Start

### Using Windows Command Prompt (cmd)

```batch
# Run the full setup and start all services
start.bat

# Install dependencies only
start.bat --deps

# Run only the Streamlit app
start.bat --streamlit

# Run in debug mode
start.bat --debug
```

### Using PowerShell

```powershell
# Run the full setup and start all services
.\start.ps1

# Install dependencies only
.\start.ps1 --deps

# Run only the Streamlit app
.\start.ps1 --streamlit

# Run in debug mode
.\start.ps1 --debug
```

### Using Python Directly

```bash
# Run the full setup and start all services
python setup.py

# Install dependencies only
python setup.py --deps

# Run only the Streamlit app
python setup.py --streamlit

# Run in debug mode
python setup.py --debug
```

## Available Options

All scripts accept the following command-line arguments:

- `--setup`: Set up directories and environment files
- `--deps`: Install dependencies
- `--db`: Set up and migrate the database
- `--api`: Run the API server only
- `--worker`: Run the Celery worker only
- `--streamlit`: Run the Streamlit app only
- `--all`: Run the full stack (API, worker, and Streamlit)
- `--debug`: Run in debug mode with hot-reloading enabled

If no arguments are provided, it defaults to running the full setup and all services.

## Services

When running the full stack, the following services will be available:

- API server: http://localhost:8000
- Streamlit app: http://localhost:8501

## Troubleshooting

### Redis Server Issues

If you encounter errors related to Redis, make sure Redis is installed and running on your system.

For Windows:
1. Download and install Redis from https://github.com/tporadowski/redis/releases
2. Run Redis server before starting the CurioScan services

For macOS:
```bash
brew install redis
brew services start redis
```

For Linux:
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

### Celery Worker Issues on Windows

On Windows, Celery might require additional configuration:
- Make sure to use `--pool=solo` when running the worker (the setup script handles this automatically)
- Ensure Redis is properly installed and running

### Database Migration Issues

If you encounter database migration errors:
1. Delete the `curioscan_test.db` file (if it exists)
2. Run `python setup.py --db` to recreate the database from scratch

## Contributing

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on our code of conduct and the process for submitting pull requests.
