from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

# Create a test PDF with text content
doc = fitz.open()  # Create new PDF
page = doc.new_page(width=595, height=842)  # A4 size

# Add some text to the page
text = """
SAMPLE DOCUMENT FOR OCR TESTING

This is a test document created to verify that the OCR pipeline
is working correctly. It contains multiple lines of text that
should be extracted by the OCR engine.

Key Information:
- Document Type: Test PDF
- Purpose: OCR Validation
- Date: 2025-09-03
- Status: Active

The quick brown fox jumps over the lazy dog.
This sentence contains all letters of the alphabet.

Additional test content:
1. First item in the list
2. Second item with numbers: 12345
3. Third item with symbols: @#$%^&*()

Contact Information:
Email: test@example.com
Phone: (555) 123-4567
Address: 123 Main St, Test City, TC 12345
"""

# Insert text into the PDF
text_rect = fitz.Rect(50, 50, 545, 792)  # Text area with margins
page.insert_textbox(text_rect, text, fontsize=12, fontname="helv")

# Save the test PDF
test_pdf_path = "test_document.pdf"
doc.save(test_pdf_path)
doc.close()

print(f"Created test PDF: {test_pdf_path}")
print("This PDF contains actual text content for OCR testing.")
