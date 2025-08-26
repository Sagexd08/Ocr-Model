# CurioScan OCR Startup Script for Windows PowerShell
# This script runs the CurioScan OCR project without Docker

Write-Host "=== CurioScan OCR Startup ===" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Using $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if setup.py exists
if (-not (Test-Path -Path "setup.py")) {
    Write-Host "Error: setup.py not found. Please run this script from the project root directory." -ForegroundColor Red
    exit 1
}

# Parse command line arguments
$setupArg = $false
$depsArg = $false
$dbArg = $false
$apiArg = $false
$workerArg = $false
$streamlitArg = $false
$allArg = $false
$debugArg = $false

foreach ($arg in $args) {
    switch ($arg) {
        "--setup" { $setupArg = $true }
        "--deps" { $depsArg = $true }
        "--db" { $dbArg = $true }
        "--api" { $apiArg = $true }
        "--worker" { $workerArg = $true }
        "--streamlit" { $streamlitArg = $true }
        "--all" { $allArg = $true }
        "--debug" { $debugArg = $true }
        default { Write-Host "Unknown argument: $arg" -ForegroundColor Yellow }
    }
}

# If no arguments provided, run everything
if (-not ($setupArg -or $depsArg -or $dbArg -or $apiArg -or $workerArg -or $streamlitArg -or $allArg)) {
    $setupArg = $true
    $depsArg = $true
    $dbArg = $true
    $allArg = $true
}

# Build the command
$cmd = "python setup.py"
if ($setupArg) { $cmd += " --setup" }
if ($depsArg) { $cmd += " --deps" }
if ($dbArg) { $cmd += " --db" }
if ($apiArg) { $cmd += " --api" }
if ($workerArg) { $cmd += " --worker" }
if ($streamlitArg) { $cmd += " --streamlit" }
if ($allArg) { $cmd += " --all" }
if ($debugArg) { $cmd += " --debug" }

# Run the command
Write-Host "Running: $cmd" -ForegroundColor Blue
Invoke-Expression $cmd
