#!/usr/bin/env python3
"""
OCR System Streamlit App Launcher
Manual launcher for the OCR Streamlit application
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def main():
    """Launch the OCR Streamlit application"""
    
    print("🚀 OCR System - Streamlit App Launcher")
    print("=" * 50)
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Check if streamlit_app.py exists
    app_file = "streamlit_app.py"
    if not os.path.exists(app_file):
        print(f"❌ Error: {app_file} not found in current directory")
        return 1
    
    print(f"✅ Found {app_file}")
    
    # Check dependencies
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas"])
    
    try:
        from worker.model_manager import ModelManager
        print("✅ OCR system components available")
    except ImportError as e:
        print(f"❌ OCR system import error: {e}")
        print("Please ensure you're in the correct directory with the OCR system files")
        return 1
    
    # Launch Streamlit
    port = 8501
    url = f"http://localhost:{port}"
    
    print(f"🌐 Starting Streamlit server on {url}")
    print("📝 Note: The server will start in a new terminal window")
    print("🔧 Use Ctrl+C in the terminal to stop the server")
    print("=" * 50)
    
    try:
        # Try to open browser after a delay
        def open_browser():
            time.sleep(3)
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
        
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", str(port),
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"🚀 Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down OCR System...")
        return 0
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("\n🔧 Manual launch instructions:")
        print(f"1. Open a terminal/command prompt")
        print(f"2. Navigate to: {script_dir}")
        print(f"3. Run: streamlit run {app_file}")
        print(f"4. Open browser to: {url}")
        return 1

if __name__ == "__main__":
    exit(main())
