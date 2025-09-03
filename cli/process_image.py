import argparse
import json
import uuid
import time
from pathlib import Path

from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from worker.document_processor import EnhancedDocumentProcessor


def main():
    parser = argparse.ArgumentParser(description="Process an image through the pipeline and export outputs")
    parser.add_argument("image", type=str, help="Path to an image file (png/jpg/tiff)")
    parser.add_argument("--mode", type=str, default="advanced", choices=["basic","standard","enhanced","advanced"], help="Processing mode")
    parser.add_argument("--profile", type=str, default="performance", choices=["default","performance","quality"], help="Pipeline profile")
    parser.add_argument("--export", type=str, nargs="*", default=["json","csv","excel","pdf"], help="Export formats")
    parser.add_argument("--export-format", type=str, default=None, help="Override Exporter default format (e.g., pdf)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    # Validate image extension
    ext = Path(args.image).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
        raise SystemExit(f"Unsupported image format: {ext}")

    model_manager = ModelManager()
    storage_manager = StorageManager()
    processor = EnhancedDocumentProcessor(model_manager=model_manager, storage_manager=storage_manager)

    job_id = str(uuid.uuid4())
    start = time.perf_counter()
    results = processor.process_document(
        job_id=job_id,
        document_path=args.image,
        params={
            "mode": args.mode,
            "profile": args.profile,
            "classify_document": True,
            "extract_tables": True,
            "extract_metadata": False,
            "export_formats": args.export,
            "output_dir": args.output_dir,
            "export_format": args.export_format,
        }
    )
    elapsed = time.perf_counter() - start

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / (Path(args.image).stem + "_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results JSON: {json_path}")

    print(f"Processing time: {elapsed:.2f}s")
    if elapsed > 120:
        print("Warning: processing exceeded 2 minutes")


if __name__ == "__main__":
    main()

