#!/usr/bin/env python3
"""
OCR System Streamlit Launcher
Starts the Streamlit web interface for the OCR system
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    
    print("🚀 Starting OCR System Web Interface...")
    print("=" * 50)
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if streamlit is available
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas"])
    
    # Launch Streamlit
    print("🌐 Launching web interface...")
    print("📍 URL: http://localhost:8501")
    print("🔧 Use Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Run streamlit with optimized settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down OCR System...")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
