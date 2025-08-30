import argparse
import json
from pathlib import Path
from statistics import mean

from worker.model_manager import ModelManager
from worker.storage_manager import StorageManager
from worker.document_processor import EnhancedDocumentProcessor


def main():
    parser = argparse.ArgumentParser(description="Calibrate OCR/region thresholds from a batch of documents")
    parser.add_argument("inputs", nargs="+", help="Paths to PDFs or images")
    parser.add_argument("--profile", type=str, default="default", choices=["default","performance","quality"], help="Pipeline profile")
    parser.add_argument("--mode", type=str, default="standard", choices=["basic","standard","enhanced","advanced"], help="Processing mode")
    parser.add_argument("--out", type=str, default="calibration_report.json", help="Output report path")
    args = parser.parse_args()

    model_manager = ModelManager()
    storage_manager = StorageManager()
    processor = EnhancedDocumentProcessor(model_manager=model_manager, storage_manager=storage_manager)

    ocr_confidences = []
    region_confidences = []
    # Histograms bins
    bins = [i/20.0 for i in range(21)]  # 0.0..1.0 step 0.05
    ocr_hist = [0 for _ in range(len(bins))]
    region_hist = [0 for _ in range(len(bins))]

    for path in args.inputs:
        try:
            job_id = Path(path).stem
            results = processor.process_document(
                job_id=job_id,
                document_path=path,
                params={
                    "mode": args.mode,
                    "classify_document": True,
                    "extract_tables": False,
                    "extract_metadata": False,
                    "profile": args.profile,
                }
            )
            # Extract confidences from the returned summary, if present
            # Ideally, the processor would return detailed pages with tokens/regions
            pages = results.get("pages", []) if isinstance(results, dict) else []
            for page in pages:
                # OCR tokens, if surfaced
                for region in page.get("text_regions", []):
                    c = region.get("confidence")
                    if isinstance(c, (int, float)):
                        ocr_confidences.append(float(c))
                # Classified regions, if surfaced
                for r in page.get("regions", []):
                    c = r.get("confidence")
                    if isinstance(c, (int, float)):
                        region_confidences.append(float(c))
        except Exception:
            continue

    # Build histograms
    def fill_hist(values, hist):
        for v in values:
            idx = min(int(round(v * 20)), 20)
            hist[idx] += 1
    fill_hist(ocr_confidences, ocr_hist)
    fill_hist(region_confidences, region_hist)

    # Placeholder computations; in real pipeline, hook into detailed results
    def percentiles(values):
        if not values:
            return {"p50": None, "p90": None, "p95": None}
        vs = sorted(values)
        def pct(p):
            k = int(round((p/100.0) * (len(vs)-1)))
            return vs[k]
        return {"p50": pct(50), "p90": pct(90), "p95": pct(95)}

    report = {
        "profile": args.profile,
        "mode": args.mode,
        "dataset_size": len(args.inputs),
        "ocr": {
            "avg_confidence": mean(ocr_confidences) if ocr_confidences else None,
            "hist_bins": [round(b,2) for b in bins],
            "hist_counts": ocr_hist,
            "percentiles": percentiles(ocr_confidences),
            "suggested_threshold": 0.5 if not ocr_confidences else max(0.3, min(0.9, percentiles(ocr_confidences)["p50"] * 0.9)),
        },
        "regions": {
            "avg_confidence": mean(region_confidences) if region_confidences else None,
            "hist_bins": [round(b,2) for b in bins],
            "hist_counts": region_hist,
            "percentiles": percentiles(region_confidences),
            "suggested_threshold": 0.4 if not region_confidences else max(0.3, min(0.9, percentiles(region_confidences)["p50"] * 0.9)),
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved calibration report: {args.out}")


if __name__ == "__main__":
    main()

