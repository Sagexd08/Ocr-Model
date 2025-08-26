@echo off
REM CurioScan OCR Startup Script for Windows
REM This script runs the CurioScan OCR project without Docker

echo === CurioScan OCR Startup ===

REM Check if Python is installed
python --version > NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Check if setup.py exists
if not exist setup.py (
    echo Error: setup.py not found. Please run this script from the project root directory.
    exit /b 1
)

REM Parse command line arguments
set SETUP=0
set DEPS=0
set DB=0
set RUN_API=0
set RUN_WORKER=0
set RUN_STREAMLIT=0
set RUN_ALL=0
set DEBUG=0

:parse_args
if "%~1"=="" goto run_commands
if /i "%~1"=="--setup" set SETUP=1
if /i "%~1"=="--deps" set DEPS=1
if /i "%~1"=="--db" set DB=1
if /i "%~1"=="--api" set RUN_API=1
if /i "%~1"=="--worker" set RUN_WORKER=1
if /i "%~1"=="--streamlit" set RUN_STREAMLIT=1
if /i "%~1"=="--all" set RUN_ALL=1
if /i "%~1"=="--debug" set DEBUG=1
shift
goto parse_args

:run_commands
REM If no arguments provided, run everything
if %SETUP%==0 if %DEPS%==0 if %DB%==0 if %RUN_API%==0 if %RUN_WORKER%==0 if %RUN_STREAMLIT%==0 if %RUN_ALL%==0 (
    set SETUP=1
    set DEPS=1
    set DB=1
    set RUN_ALL=1
)

REM Build the command
set CMD=python setup.py
if %SETUP%==1 set CMD=%CMD% --setup
if %DEPS%==1 set CMD=%CMD% --deps
if %DB%==1 set CMD=%CMD% --db
if %RUN_API%==1 set CMD=%CMD% --api
if %RUN_WORKER%==1 set CMD=%CMD% --worker
if %RUN_STREAMLIT%==1 set CMD=%CMD% --streamlit
if %RUN_ALL%==1 set CMD=%CMD% --all
if %DEBUG%==1 set CMD=%CMD% --debug

REM Run the command
%CMD%
