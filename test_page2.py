import fitz
import tempfile
import cv2
import numpy as np
from models.ocr_models import PaddleOCRAdapter

# Test page 2 which might have more content
doc = fitz.open(r'C:\Users\sohom\OneDrive\Desktop\165_Form20.pdf')

# Check multiple pages for content
for page_num in range(min(5, len(doc))):
    page = doc[page_num]
    
    # Check native text
    native_text = page.get_text().strip()
    
    # Create image
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes('png')
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check image properties
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    print(f"Page {page_num + 1}:")
    print(f"  - Native text: {len(native_text)} chars")
    print(f"  - Mean brightness: {mean_brightness:.1f}")
    
    if len(native_text) > 0:
        print(f"  - Sample text: {native_text[:100]}...")
    
    # If page looks promising, test OCR
    if mean_brightness < 240 and len(native_text) == 0:  # Scanned page with content
        print(f"  - Testing OCR on page {page_num + 1}...")
        
        # Save temp image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_path = temp.name
        cv2.imwrite(temp_path, img)
        
        try:
            adapter = PaddleOCRAdapter(lang='en')
            result = adapter.extract_text(temp_path)
            print(f"  - OCR tokens: {len(result.tokens)}")
            if result.tokens:
                sample_text = ' '.join([t.text for t in result.tokens[:10]])
                print(f"  - Sample OCR: {sample_text[:100]}...")
                break  # Found content, stop here
        except Exception as e:
            print(f"  - OCR error: {e}")
        
        import os
        os.unlink(temp_path)

doc.close()
