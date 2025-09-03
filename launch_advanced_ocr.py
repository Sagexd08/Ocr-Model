#!/usr/bin/env python3
"""
Advanced OCR System Launcher
Comprehensive launcher for the enterprise-grade OCR Streamlit application
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def main():
    """Launch the Advanced OCR Streamlit application"""
    
    print("🚀 Advanced OCR System - Enterprise Launcher")
    print("=" * 60)
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Check if the app file exists
    app_file = "advanced_ocr_app.py"
    if not os.path.exists(app_file):
        print(f"❌ Error: {app_file} not found in current directory")
        return 1
    
    print(f"✅ Found {app_file}")
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas"])
    
    try:
        import plotly
        print(f"✅ Plotly version: {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "plotly"])
    
    try:
        from worker.model_manager import ModelManager
        print("✅ OCR system components available")
    except ImportError as e:
        print(f"⚠️  OCR system import warning: {e}")
        print("The app will still launch but OCR functionality may be limited")
    
    # Launch configuration
    port = 8505
    url = f"http://localhost:{port}"
    
    print(f"🌐 Starting Advanced OCR System on {url}")
    print("📝 Features available:")
    print("   • Enterprise-grade document processing")
    print("   • Advanced analytics and visualizations")
    print("   • Real-time performance monitoring")
    print("   • Comprehensive export options")
    print("   • Multi-format document support")
    print("=" * 60)
    
    try:
        # Open browser after a delay
        def open_browser():
            time.sleep(5)
            try:
                webbrowser.open(url)
                print(f"🌐 Opened browser to {url}")
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
                print(f"Please manually open: {url}")
        
        # Start browser opener in background
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Launch Streamlit with optimized settings
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", str(port),
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--server.maxUploadSize", "200"
        ]
        
        print(f"🚀 Executing: {' '.join(cmd)}")
        print("🔧 Use Ctrl+C to stop the server")
        print("=" * 60)
        
        result = subprocess.run(cmd)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Advanced OCR System...")
        return 0
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("\n🔧 Manual launch instructions:")
        print(f"1. Open a terminal/command prompt")
        print(f"2. Navigate to: {script_dir}")
        print(f"3. Run: streamlit run {app_file} --server.port {port}")
        print(f"4. Open browser to: {url}")
        return 1

if __name__ == "__main__":
    exit(main())
