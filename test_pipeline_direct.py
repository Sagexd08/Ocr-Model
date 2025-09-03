from worker.pipeline.processors.pdf_processor import PDFProcessor
from worker.pipeline.processors.advanced_ocr import AdvancedOCRProcessor
from worker.pipeline.processors.exporter import Exporter
from worker.types import Document

# Test the pipeline directly
print("Testing pipeline components directly...")

# Create document
doc = Document(id="test-doc")
doc.metadata["source_path"] = "test_document.pdf"
doc.metadata["doc_type"] = "pdf"

print(f"Initial document: {len(doc.pages)} pages")

# Test PDFProcessor
print("\n1. Testing PDFProcessor...")
pdf_proc = PDFProcessor()
setattr(pdf_proc, 'max_pages', 1)
doc = pdf_proc.process(doc)
print(f"After PDFProcessor: {len(doc.pages)} pages")

if doc.pages:
    page = doc.pages[0]
    print(f"  Page 1: {len(page.tokens)} tokens, image: {page.image is not None}")
    if page.tokens:
        print(f"  First token: '{page.tokens[0].text}' (conf: {page.tokens[0].confidence:.2f})")

# Test OCR Processor
print("\n2. Testing AdvancedOCRProcessor...")
ocr_proc = AdvancedOCRProcessor()
doc = ocr_proc.process(doc)
print(f"After OCR: {len(doc.pages)} pages")

if doc.pages:
    page = doc.pages[0]
    print(f"  Page 1: {len(page.tokens)} tokens")
    if page.tokens:
        print(f"  First token: '{page.tokens[0].text}' (conf: {page.tokens[0].confidence:.2f})")
        sample_text = ' '.join([t.text for t in page.tokens[:10]])
        print(f"  Sample text: {sample_text[:100]}...")

# Test Exporter
print("\n3. Testing Exporter...")
exporter = Exporter({"default_format": "json", "output_dir": "output"})
doc = exporter.process(doc)
print(f"After Exporter: {len(doc.pages)} pages")

print("\nPipeline test complete!")
