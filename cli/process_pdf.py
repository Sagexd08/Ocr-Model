import argparse
from pathlib import Path
import json
import time

from worker.document_processor import EnhancedDocumentProcessor
from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import uuid


def main():
    parser = argparse.ArgumentParser(description="Process a PDF through the full pipeline and export outputs")
    parser.add_argument("pdf", type=str, help="Path to PDF file")
    parser.add_argument("--mode", type=str, default="advanced", choices=["basic","standard","enhanced","advanced"], help="Processing mode")
    parser.add_argument("--profile", type=str, default="performance", choices=["default","performance","quality"], help="Pipeline profile")
    parser.add_argument("--export", type=str, nargs="*", default=["json","csv","excel","pdf"], help="Export formats")
    parser.add_argument("--export-format", type=str, default=None, help="Override Exporter default format (e.g., pdf)")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional max pages to process")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    # Build processor with managers
    model_manager = ModelManager()
    storage_manager = StorageManager()
    processor = EnhancedDocumentProcessor(model_manager=model_manager, storage_manager=storage_manager)
    if args.max_pages is not None:
        # PDFProcessor reads this attr if present to limit processing
        # (see worker.pipeline.processors.pdf_processor: process_pdf)
        processor.max_pages = args.max_pages

    job_id = str(uuid.uuid4())
    start = time.perf_counter()
    results = processor.process_document(
        job_id=job_id,
        document_path=args.pdf,
        params={
            "mode": args.mode,
            "extract_tables": True,
            "classify_document": True,
            "extract_metadata": True,
            "export_formats": args.export,
            "output_dir": args.output_dir,
            "profile": args.profile,
            "export_format": args.export_format,
            "max_pages": args.max_pages,
        }
    )
    elapsed = time.perf_counter() - start

    # Save the structured JSON result as well
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / (Path(args.pdf).stem + "_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results JSON: {json_path}")

    # Show processing time
    print(f"Processing time: {elapsed:.2f}s")
    if elapsed > 120:
        print("Warning: processing exceeded 2 minutes")


if __name__ == "__main__":
    main()

