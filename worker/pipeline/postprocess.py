"""
Postprocessing utilities to assemble final rows and compute confidences.
"""

from typing import Dict, Any, List
from worker.types import PostprocessResultTD, RowTD, OCRResultsTD, TableResultsTD, ClassificationTD


def assemble_results(ocr_results: OCRResultsTD, table_results: TableResultsTD, classification_result: ClassificationTD) -> PostprocessResultTD:
    rows: List[RowTD] = []
    for page in ocr_results.get("pages", []):
        for region in page.get("regions", []):
            text = " ".join(tok.get("text", "") for tok in region.get("tokens", []))
            if text.strip():
                rows.append({
                    "row_id": f"row_{len(rows)+1}",
                    "page": page.get("page_number", 1),
                    "region_id": region.get("id", "text"),
                    "bbox": region.get("bbox", [0, 0, 0, 0]),
                    "columns": {"text": text},
                    "provenance": {
                        "token_ids": [tok.get("token_id") for tok in region.get("tokens", []) if tok.get("token_id") is not None],
                        "confidence": min((tok.get("confidence", 1.0) for tok in region.get("tokens", [])), default=1.0)
                    },
                    "needs_review": False
                })
    confidence_score: float = float(ocr_results.get("avg_confidence", 0.9) or 0.9)
    return {
        "rows": rows,
        "metadata": {
            "render_type": classification_result.get("render_type"),
            "total_pages": len(ocr_results.get("pages", [])),
            "confidence_threshold": classification_result.get("confidence_threshold", 0.8),
        },
        "confidence_score": confidence_score,
        "render_type": classification_result.get("render_type"),
    }

