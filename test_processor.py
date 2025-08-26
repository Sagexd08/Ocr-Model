#!/usr/bin/env python3
"""
Test script for the enhanced document processing pipeline.
This script allows for quick testing of document processing without going through the API.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager
from worker.enhanced_document_processor import EnhancedDocumentProcessor
from worker.types import ProcessingMode, DocumentType

def process_document(input_path: str, output_dir: str, config: Dict[str, Any] = None) -> None:
    """
    Process a document using the enhanced document processor.
    
    Args:
        input_path: Path to input document
        output_dir: Directory to save results
        config: Processing configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize managers and processor
    model_manager = ModelManager(models_path=os.environ.get("MODELS_PATH", "models"))
    storage_manager = StorageManager(
        input_path=os.path.dirname(input_path),
        output_path=output_dir
    )
    
    processor = EnhancedDocumentProcessor(model_manager, storage_manager)
    
    # Generate job ID from filename
    job_id = f"test_{int(time.time())}_{os.path.basename(input_path)}"
    
    # Default processing parameters
    params = {
        "processing_mode": ProcessingMode.AUTO,
        "extract_tables": True,
        "extract_forms": True,
        "output_formats": ["json", "text"],
    }
    
    # Update with user config
    if config:
        params.update(config)
    
    print(f"Processing document: {input_path}")
    print(f"Job ID: {job_id}")
    print(f"Parameters: {params}")
    
    # Process the document
    start_time = time.time()
    result = processor.process_document(job_id, input_path, params)
    end_time = time.time()
    
    # Print results
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Status: {result['status']}")
    
    if result['status'] == "COMPLETED":
        print(f"Document type: {result.get('document_type')}")
        print(f"Page count: {result.get('page_count', 0)}")
        print(f"Results saved to: {result.get('result_path')}")
    else:
        print(f"Error: {result.get('error')}")

def main():
    parser = argparse.ArgumentParser(description='Test OCR document processing')
    parser.add_argument('input', help='Path to input document')
    parser.add_argument('-o', '--output', help='Output directory', default='output')
    parser.add_argument('-m', '--mode', help='Processing mode (auto, ocr, native, legacy)', default='auto')
    parser.add_argument('-t', '--tables', help='Extract tables', action='store_true')
    parser.add_argument('-f', '--forms', help='Extract forms', action='store_true')
    parser.add_argument('-c', '--config', help='JSON config file')
    
    args = parser.parse_args()
    
    # Load config from file if specified
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Build config from command line args
        config = {
            "processing_mode": args.mode.upper(),
            "extract_tables": args.tables,
            "extract_forms": args.forms,
        }
    
    process_document(args.input, args.output, config)

if __name__ == "__main__":
    main()
