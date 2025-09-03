from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from models.ocr_models import PaddleOCRAdapter

# Create a simple test image with text
img = Image.new('RGB', (800, 200), color='white')
draw = ImageDraw.Draw(img)

# Use default font
try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

# Draw some text
text = "Hello World! This is a test document for OCR processing."
draw.text((50, 50), text, fill='black', font=font)

# Save temp image
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
    temp_path = temp.name
img.save(temp_path)

print(f"Created test image: {temp_path}")
print(f"Test text: {text}")

# Test OCR
try:
    adapter = PaddleOCRAdapter(lang='en')
    result = adapter.extract_text(temp_path)
    print(f"OCR tokens found: {len(result.tokens)}")
    
    if result.tokens:
        print("OCR Results:")
        for i, token in enumerate(result.tokens):
            print(f"  {i+1}: '{token.text}' (conf: {token.confidence:.2f})")
        
        # Reconstruct text
        ocr_text = ' '.join([t.text for t in result.tokens])
        print(f"Reconstructed text: {ocr_text}")
    else:
        print("No tokens found - OCR may not be working properly")
        
except Exception as e:
    print(f"OCR error: {e}")
    import traceback
    traceback.print_exc()

# Clean up
os.unlink(temp_path)
