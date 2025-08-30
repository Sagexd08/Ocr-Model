# Installing PaddleOCR (optional but recommended)

PaddleOCR provides robust text detection and recognition that significantly improves accuracy on scanned and rotated documents. We include PaddleOCR as an optional dependency. Follow the steps below for your environment.

## 1) Choose paddlepaddle build

PaddleOCR depends on paddlepaddle. You must install a CPU or GPU wheel that matches your OS and Python.

- CPU (Windows/Linux/Mac):
  - python -m pip install --upgrade pip
  - python -m pip install paddlepaddle==2.5.2 -i https://mirror.baidu.com/pypi/simple

- GPU (Windows/Linux):
  - Ensure CUDA/CUDNN drivers match the wheel version
  - See: https://www.paddlepaddle.org.cn/install/quick

If the default PyPI mirror fails on your network, use the official mirror links above.

## 2) Install PaddleOCR

- python -m pip install paddleocr==2.7.0

Note: On Windows, Visual C++ Build Tools may be required for some extras. If you hit build errors, install "Desktop development with C++" workload via Visual Studio Installer.

## 3) Verify install

Run the following in a Python shell:

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
print('PaddleOCR ready')
```

The first call may download model weights.

## 4) Using in this project

- We added paddleocr to requirements.txt and worker/requirements.txt, but paddlepaddle itself must be installed separately per your platform.
- Our code loads PaddleOCR opportunistically; if not installed, the system falls back to Tesseract/EasyOCR.

## 5) Troubleshooting

- ImportError: No module named 'paddleocr':
  - Install with `pip install paddleocr` and ensure paddlepaddle is installed.
- GPU errors / CUDA version mismatch:
  - Verify your installed paddlepaddle-gpu wheel matches your CUDA version.
- Large model downloads behind proxy:
  - Set HTTP/HTTPS proxy environment variables, or pre-download models and set cache dirs.

