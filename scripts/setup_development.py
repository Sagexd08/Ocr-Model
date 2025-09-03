#!/usr/bin/env python3
"""
Development Environment Setup Script
Automates the setup of development environment for Enterprise OCR System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, check=True):
    """Run a command and return the result"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    print("\n🔧 Setting up virtual environment...")
    
    if not os.path.exists("venv"):
        run_command(f"{sys.executable} -m venv venv")
    
    # Activation command varies by platform
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print(f"Virtual environment created. Activate with: {activate_cmd}")
    return pip_cmd

def install_dependencies(pip_cmd):
    """Install all dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    run_command(f"{pip_cmd} install --upgrade pip")
    run_command(f"{pip_cmd} install -r requirements.txt")
    
    # Streamlit dependencies
    if os.path.exists("requirements_streamlit.txt"):
        run_command(f"{pip_cmd} install -r requirements_streamlit.txt")
    
    # Development dependencies
    if os.path.exists("requirements-dev.txt"):
        run_command(f"{pip_cmd} install -r requirements-dev.txt")

def setup_pre_commit():
    """Setup pre-commit hooks"""
    print("\n🔗 Setting up pre-commit hooks...")
    
    if os.path.exists(".pre-commit-config.yaml"):
        run_command("pre-commit install")
    else:
        print("No pre-commit configuration found, skipping...")

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "data/storage/input",
        "data/storage/output", 
        "data/storage/cache",
        "logs",
        "output",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def check_system_dependencies():
    """Check for system dependencies"""
    print("\n🔍 Checking system dependencies...")
    
    # Check for required system libraries
    system = platform.system()
    
    if system == "Linux":
        print("On Linux, ensure these packages are installed:")
        print("  sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1")
    elif system == "Darwin":  # macOS
        print("On macOS, ensure Xcode command line tools are installed:")
        print("  xcode-select --install")
    elif system == "Windows":
        print("On Windows, ensure Visual C++ redistributables are installed")

def run_tests():
    """Run basic tests to verify setup"""
    print("\n🧪 Running basic tests...")
    
    # Test Python imports
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} imported successfully")
    except ImportError:
        print("❌ Streamlit import failed")
    
    try:
        import paddleocr
        print("✅ PaddleOCR imported successfully")
    except ImportError:
        print("❌ PaddleOCR import failed")
    
    # Run pytest if available
    if os.path.exists("tests"):
        result = run_command("python -m pytest tests/ --tb=short", check=False)
        if result and result.returncode == 0:
            print("✅ Tests passed")
        else:
            print("⚠️  Some tests failed or pytest not available")

def main():
    """Main setup function"""
    print("🚀 Enterprise OCR System - Development Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    pip_cmd = setup_virtual_environment()
    
    # Install dependencies
    install_dependencies(pip_cmd)
    
    # Setup pre-commit
    setup_pre_commit()
    
    # Create directories
    create_directories()
    
    # Check system dependencies
    check_system_dependencies()
    
    # Run tests
    run_tests()
    
    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Launch the application:")
    print("   python launch_advanced_ocr.py")
    print("3. Open browser to: http://localhost:8505")

if __name__ == "__main__":
    main()
