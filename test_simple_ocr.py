#!/usr/bin/env python3

# Simple test to verify OCR is working end-to-end
import time
from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager
from worker.document_processor import EnhancedDocumentProcessor
from worker.types import ProcessingMode

def main():
    print("=== OCR SYSTEM VALIDATION ===")
    
    # Initialize components
    print("1. Initializing components...")
    model_manager = ModelManager()
    storage_manager = StorageManager()
    processor = EnhancedDocumentProcessor(model_manager=model_manager, storage_manager=storage_manager)
    print(f"   Processor class: {type(processor)}")
    print(f"   Processor methods: {[m for m in dir(processor) if 'process' in m.lower()]}")
    
    # Test document
    test_pdf = "test_document.pdf"
    print(f"2. Processing document: {test_pdf}")
    
    # Process with minimal settings
    start_time = time.time()
    print(f"   Calling process_document with job_id='test-validation', document_path='{test_pdf}'")
    try:
        result = processor.process_document(
            job_id="test-validation",
            document_path=test_pdf,
            params={
                "mode": "advanced",
                "max_pages": 1,
                "profile": "performance",
                "allow_cache": False,  # Disable cache
                "export_format": "json"
            }
        )
        print(f"   process_document returned successfully")
    except Exception as e:
        print(f"   ERROR in process_document: {e}")
        import traceback
        traceback.print_exc()
        return False
    processing_time = time.time() - start_time
    
    print(f"3. Processing completed in {processing_time:.2f}s")
    print(f"   Status: {result.get('status')}")
    print(f"   Message: {result.get('message')}")
    
    # Check results
    result_path = result.get('result_path')
    if result_path:
        print(f"4. Checking results: {result_path}")
        try:
            import json
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pages = data.get('pages', [])
            print(f"   Pages found: {len(pages)}")
            
            if pages:
                page = pages[0]
                tokens = page.get('tokens', [])
                print(f"   Tokens on page 1: {len(tokens)}")
                
                if tokens:
                    print("   SUCCESS: OCR extracted text!")
                    sample_text = ' '.join([t.get('text', '') for t in tokens[:5]])
                    print(f"   Sample: {sample_text[:100]}...")
                    return True
                else:
                    print("   ISSUE: No tokens found")
            else:
                print("   ISSUE: No pages found")
        except Exception as e:
            print(f"   ERROR reading results: {e}")
    
    print("   FAILED: OCR system not working correctly")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
