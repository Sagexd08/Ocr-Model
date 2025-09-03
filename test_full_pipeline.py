from worker.document_processor import EnhancedDocumentProcessor
from worker.types import ProcessingMode

# Test the full document processor
print("Testing full document processor...")

processor = EnhancedDocumentProcessor()

# Process the test document
result = processor.process_document(
    document_path="test_document.pdf",
    processing_mode=ProcessingMode.ADVANCED,
    params={
        "max_pages": 1,
        "profile": "performance",
        "export_format": "json"
    },
    output_format="json"
)

print(f"Processing result:")
print(f"- Status: {result.get('status')}")
print(f"- Message: {result.get('message')}")
print(f"- Processing time: {result.get('processing_duration', 0):.2f}s")
print(f"- Cache used: {result.get('cache', False)}")

# Check if result_path exists and load it
result_path = result.get('result_path')
if result_path:
    print(f"- Result path: {result_path}")
    try:
        import json
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"- Pages in result: {len(data.get('pages', []))}")
        if data.get('pages'):
            page = data['pages'][0]
            tokens = page.get('tokens', [])
            print(f"- Tokens on page 1: {len(tokens)}")
            if tokens:
                sample_text = ' '.join([t.get('text', '') for t in tokens[:5]])
                print(f"- Sample text: {sample_text[:100]}...")
    except Exception as e:
        print(f"- Error reading result: {e}")

# Check summary
summary = result.get('summary', {})
print(f"- Summary word count: {summary.get('word_count', 0)}")
print(f"- Summary page count: {summary.get('page_count', 0)}")

print("\nFull pipeline test complete!")
