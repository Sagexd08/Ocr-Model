import fitz
import tempfile
import cv2
import numpy as np
from PIL import Image

# Debug the PDF page
doc = fitz.open(r'C:\Users\sohom\OneDrive\Desktop\165_Form20.pdf')
page = doc[0]

print("PDF Info:")
print(f"- Page count: {len(doc)}")
print(f"- Page size: {page.rect}")

# Check if page has native text
text_dict = page.get_text("dict")
blocks = text_dict.get("blocks", [])
print(f"- Native text blocks: {len(blocks)}")

# Check for text spans
span_count = 0
for block in blocks:
    if "lines" in block:
        for line in block["lines"]:
            span_count += len(line.get("spans", []))
print(f"- Text spans: {span_count}")

# Get native text
native_text = page.get_text()
print(f"- Native text length: {len(native_text)}")
if native_text.strip():
    print(f"- Native text sample: {native_text[:200]}...")

# Create image and save for inspection
pix = page.get_pixmap(dpi=150)  # Lower DPI for faster processing
img_bytes = pix.tobytes('png')
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Save image for manual inspection
cv2.imwrite('debug_page.png', img)
print(f"- Saved page image as debug_page.png ({img.shape})")

# Check image properties
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
unique_colors = len(np.unique(gray))
print(f"- Unique gray levels: {unique_colors}")
print(f"- Image mean brightness: {np.mean(gray):.1f}")

# Simple text detection using contours
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
text_like_contours = [c for c in contours if cv2.contourArea(c) > 50 and cv2.contourArea(c) < 10000]
print(f"- Text-like contours: {len(text_like_contours)}")

doc.close()
