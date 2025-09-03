# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### 🚀 Installation Issues

#### Problem: PaddleOCR Installation Fails
```bash
ERROR: Failed building wheel for paddlepaddle
```

**Solution:**
```bash
# For Windows
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple/

# For macOS (Intel)
pip install paddlepaddle

# For macOS (Apple Silicon)
pip install paddlepaddle --index-url https://pypi.org/simple/

# For Linux
pip install paddlepaddle-gpu  # If you have CUDA
pip install paddlepaddle      # CPU only
```

#### Problem: Streamlit Import Error
```bash
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install streamlit plotly pandas
# Or use the requirements file
pip install -r requirements_streamlit.txt
```

#### Problem: Memory Error During Installation
```bash
MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Increase virtual memory or use smaller batch sizes
export PYTHONHASHSEED=0
pip install --no-cache-dir -r requirements.txt
```

### 🔧 Runtime Issues

#### Problem: OCR System Initialization Fails
```bash
ERROR: Failed to initialize OCR system
```

**Solutions:**
1. **Check available memory:**
   ```bash
   # Ensure at least 4GB RAM available
   free -h  # Linux
   wmic OS get TotalVisibleMemorySize /value  # Windows
   ```

2. **Verify model download:**
   ```bash
   # Models should download to ~/.paddleocr/
   ls ~/.paddleocr/whl/
   ```

3. **Check internet connectivity:**
   ```bash
   # Models need to download on first run
   ping github.com
   ```

#### Problem: Slow Processing Performance
```bash
Processing taking longer than expected
```

**Solutions:**
1. **Use Performance Profile:**
   ```python
   # In web interface, select "Performance" profile
   # Or in CLI:
   python -m cli.process_pdf document.pdf --profile performance
   ```

2. **Limit page processing:**
   ```bash
   python -m cli.process_pdf document.pdf --max-pages 5
   ```

3. **Check system resources:**
   ```bash
   # Monitor CPU and memory usage
   htop  # Linux
   Task Manager  # Windows
   ```

#### Problem: Low OCR Accuracy
```bash
Confidence scores below 70%
```

**Solutions:**
1. **Use Quality Profile:**
   ```python
   # Switch to quality profile for better accuracy
   params = {"profile": "quality", "mode": "advanced"}
   ```

2. **Check image quality:**
   - Ensure images are high resolution (300+ DPI)
   - Verify good contrast and lighting
   - Remove noise and artifacts

3. **Preprocess images:**
   ```python
   import cv2
   
   # Enhance image quality
   image = cv2.imread('document.jpg')
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   denoised = cv2.fastNlMeansDenoising(gray)
   ```

### 🌐 Web Interface Issues

#### Problem: Streamlit App Won't Start
```bash
streamlit run advanced_ocr_app.py
# No response or error
```

**Solutions:**
1. **Check port availability:**
   ```bash
   # Check if port 8505 is in use
   netstat -an | grep 8505  # Linux/macOS
   netstat -an | findstr 8505  # Windows
   ```

2. **Try different port:**
   ```bash
   streamlit run advanced_ocr_app.py --server.port 8506
   ```

3. **Check firewall settings:**
   ```bash
   # Allow port through firewall
   sudo ufw allow 8505  # Linux
   # Windows: Add rule in Windows Firewall
   ```

#### Problem: File Upload Fails
```bash
File upload error: File too large
```

**Solutions:**
1. **Increase upload limit:**
   ```toml
   # .streamlit/config.toml
   [server]
   maxUploadSize = 500  # MB
   ```

2. **Compress large files:**
   ```bash
   # Reduce PDF size
   gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
      -dNOPAUSE -dQUIET -dBATCH -sOutputFile=compressed.pdf input.pdf
   ```

#### Problem: Browser Compatibility Issues
```bash
Interface not displaying correctly
```

**Solutions:**
1. **Use supported browsers:**
   - Chrome 80+
   - Firefox 75+
   - Safari 13+
   - Edge 80+

2. **Clear browser cache:**
   ```bash
   # Hard refresh
   Ctrl+F5 (Windows/Linux)
   Cmd+Shift+R (macOS)
   ```

### 📊 Performance Issues

#### Problem: High Memory Usage
```bash
System running out of memory
```

**Solutions:**
1. **Reduce batch size:**
   ```python
   # Process fewer pages at once
   params = {"max_pages": 1}
   ```

2. **Enable garbage collection:**
   ```python
   import gc
   gc.collect()  # Force garbage collection
   ```

3. **Monitor memory usage:**
   ```python
   import psutil
   print(f"Memory usage: {psutil.virtual_memory().percent}%")
   ```

#### Problem: CPU Bottleneck
```bash
High CPU usage, slow processing
```

**Solutions:**
1. **Optimize threading:**
   ```python
   # Limit CPU cores used
   import os
   os.environ['OMP_NUM_THREADS'] = '4'
   ```

2. **Use GPU acceleration (if available):**
   ```bash
   # Install GPU version
   pip install paddlepaddle-gpu
   ```

### 🔌 API Issues

#### Problem: API Connection Refused
```bash
Connection refused to localhost:8000
```

**Solutions:**
1. **Start API server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Check API health:**
   ```bash
   curl http://localhost:8000/health
   ```

#### Problem: API Timeout
```bash
Request timeout after 30 seconds
```

**Solutions:**
1. **Increase timeout:**
   ```python
   import requests
   response = requests.post(url, files=files, timeout=300)  # 5 minutes
   ```

2. **Use async processing:**
   ```python
   # Upload and poll for results
   job_id = upload_document(file)
   while True:
       status = check_status(job_id)
       if status['status'] == 'completed':
           break
       time.sleep(5)
   ```

### 🐳 Docker Issues

#### Problem: Docker Build Fails
```bash
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solutions:**
1. **Use multi-stage build:**
   ```dockerfile
   FROM python:3.9-slim as builder
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt
   
   FROM python:3.9-slim
   COPY --from=builder /root/.local /root/.local
   ```

2. **Increase Docker memory:**
   ```bash
   # Docker Desktop: Settings > Resources > Memory > 8GB
   ```

#### Problem: Container Out of Memory
```bash
Container killed due to memory limit
```

**Solutions:**
1. **Increase memory limit:**
   ```yaml
   # docker-compose.yml
   services:
     ocr-app:
       mem_limit: 8g
   ```

2. **Optimize container:**
   ```dockerfile
   # Use slim base image
   FROM python:3.9-slim
   
   # Remove unnecessary packages
   RUN apt-get autoremove -y && apt-get clean
   ```

### 📝 Logging and Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

#### Check System Logs
```bash
# Application logs
tail -f logs/ocr_system.log

# System logs
journalctl -u ocr-service -f  # Linux systemd
```

#### Performance Profiling
```python
import cProfile
import pstats

# Profile code execution
cProfile.run('process_document()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

### 🆘 Getting Help

#### Collect System Information
```bash
# System info script
python -c "
import sys, platform, psutil
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'CPU: {psutil.cpu_count()} cores')
print(f'Memory: {psutil.virtual_memory().total // (1024**3)} GB')
print(f'Disk: {psutil.disk_usage(\"/\").free // (1024**3)} GB free')
"
```

#### Create Bug Report
Include the following information:
1. System specifications
2. Python version and dependencies
3. Error messages and stack traces
4. Steps to reproduce
5. Sample files (if possible)

#### Contact Support
- **GitHub Issues**: [Report bugs](https://github.com/Sagexd08/Ocr-Model/issues)
- **Discussions**: [Ask questions](https://github.com/Sagexd08/Ocr-Model/discussions)
- **Email**: support@ocr-system.com

### 🔧 Quick Fixes

#### Reset System
```bash
# Clear cache and restart
rm -rf ~/.paddleocr/
rm -rf data/storage/cache/
python launch_advanced_ocr.py
```

#### Reinstall Dependencies
```bash
# Clean reinstall
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

#### Factory Reset
```bash
# Complete reset (WARNING: Deletes all data)
rm -rf data/
rm -rf output/
rm -rf logs/
git clean -fdx
pip install -r requirements.txt
```
