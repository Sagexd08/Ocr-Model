from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from paddleocr import PaddleOCR

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

# Test raw PaddleOCR
try:
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    # Try different methods
    print("\nTesting predict method:")
    try:
        result = ocr.predict(temp_path)
        print(f"Predict result type: {type(result)}")
        print(f"Predict result: {result}")
    except Exception as e:
        print(f"Predict failed: {e}")
    
    print("\nTesting ocr method:")
    try:
        result = ocr.ocr(temp_path)
        print(f"OCR result type: {type(result)}")
        print(f"OCR result: {result}")
        
        if result and len(result) > 0:
            print(f"First batch: {result[0]}")
            if result[0] and len(result[0]) > 0:
                print(f"First item: {result[0][0]}")
    except Exception as e:
        print(f"OCR failed: {e}")
        
except Exception as e:
    print(f"PaddleOCR initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up
os.unlink(temp_path)
