import fitz
import tempfile
import cv2
import numpy as np
from models.ocr_models import PaddleOCRAdapter

# Test updated PaddleOCR adapter
doc = fitz.open(r'C:\Users\sohom\OneDrive\Desktop\165_Form20.pdf')
page = doc[0]
pix = page.get_pixmap(dpi=300)
img_bytes = pix.tobytes('png')
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Save temp image
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
    temp_path = temp.name
cv2.imwrite(temp_path, img)

print('Testing updated PaddleOCR adapter...')
try:
    adapter = PaddleOCRAdapter(lang='en')
    result = adapter.extract_text(temp_path)
    print('Tokens found:', len(result.tokens))
    print('Model name:', result.model_name)
    if result.tokens:
        print('First token:', result.tokens[0].text, 'confidence:', result.tokens[0].confidence)
        print('First 5 tokens:')
        for i, token in enumerate(result.tokens[:5]):
            print(f'  {i+1}: "{token.text}" (conf: {token.confidence:.2f})')
        
        # Show sample text
        sample_text = ' '.join([t.text for t in result.tokens[:20]])
        print(f'Sample text: {sample_text[:200]}...')
except Exception as e:
    print('Adapter error:', e)
    import traceback
    traceback.print_exc()

import os
os.unlink(temp_path)
doc.close()
