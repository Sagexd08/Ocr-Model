"""
Results download endpoints for CurioScan API.
"""

import logging
import json
import csv
import io
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Response
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
import pandas as pd

from ..config import get_settings
from ..database import get_db
from ..models import Job, JobStatus
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()
from pathlib import Path



@router.get("/result/{job_id}")
async def download_result(
    job_id: str,
    format: str = "json",  # json, csv, xlsx
    include_provenance: bool = True,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Download processing results in various formats.

    This endpoint first attempts to serve exports written by the background
    processor from local storage (./data/storage/output/<job_id>.*). If not
    found, it falls back to database-backed exports when available.
    """
    try:
        fmt = format.lower()
        base_path = Path("./data/storage/output") / job_id
        json_path = base_path.with_suffix(".json")
        csv_path = base_path.with_suffix(".csv")
        xlsx_path = base_path.with_suffix(".xlsx")

        # Serve directly from filesystem if present
        if fmt == "json" and json_path.exists():
            return FileResponse(
                path=str(json_path),
                media_type="application/json",
                filename=f"{job_id}_result.json",
            )
        if fmt == "csv" and csv_path.exists():
            return FileResponse(
                path=str(csv_path),
                media_type="text/csv",
                filename=f"{job_id}_result.csv",
            )
        if fmt == "xlsx" and xlsx_path.exists():
            return FileResponse(
                path=str(xlsx_path),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=f"{job_id}_result.xlsx",
            )

        # If JSON exists, synthesize requested export on-the-fly
        if json_path.exists():
            import json as _json
            with open(json_path, "r", encoding="utf-8") as f:
                results = _json.load(f)

            if fmt == "json":
                # Return normalized JSON with provenance untouched
                return Response(
                    content=_json.dumps(results, indent=2, ensure_ascii=False),
                    media_type="application/json",
                    headers={
                        "Content-Disposition": f"attachment; filename={job_id}_result.json"
                    },
                )
            elif fmt == "csv":
                csv_bytes = _results_to_csv_bytes(results, include_provenance)
                return Response(
                    content=csv_bytes,
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename={job_id}_result.csv"
                    },
                )
            elif fmt == "xlsx":
                xlsx_bytes = _results_to_xlsx_bytes(results, include_provenance)
                return StreamingResponse(
                    io.BytesIO(xlsx_bytes),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={
                        "Content-Disposition": f"attachment; filename={job_id}_result.xlsx"
                    },
                )

        # Fallback: try DB-backed export if job/results tables are populated
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed (status: {job.status})")
        extraction_results = getattr(job, "extraction_results", None)
        if not extraction_results:
            raise HTTPException(status_code=404, detail=f"No results found for job {job_id}")

        if fmt == "json":
            return await _export_json(job, extraction_results, include_provenance)
        elif fmt == "csv":
            return await _export_csv(job, extraction_results, include_provenance)
        elif fmt == "xlsx":
            return await _export_xlsx(job, extraction_results, include_provenance)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Supported: json, csv, xlsx")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download results for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download results: {str(e)}")


async def _export_json(job: Job, extraction_results: list, include_provenance: bool):
    """Export results as JSON."""
    rows = []

    for result in extraction_results:
        row_data = {
            "row_id": result.row_id,
            "page": result.page,
            "region_id": result.region_id,
            "bbox": [result.bbox_x1, result.bbox_y1, result.bbox_x2, result.bbox_y2],
            "columns": result.columns_data,
            "needs_review": result.needs_review
        }

        if include_provenance:
            row_data["provenance"] = {
                "file": result.source_file,
                "page": result.source_page,
                "bbox": [result.source_bbox_x1, result.source_bbox_y1,
                        result.source_bbox_x2, result.source_bbox_y2],
                "token_ids": result.token_ids,
                "confidence": result.confidence
            }

        rows.append(row_data)

    result_data = {
        "rows": rows,
        "metadata": {
            "job_id": job.job_id,
            "file_name": job.file_name,
            "render_type": job.render_type,
            "processing_time": (job.completed_at - job.created_at).total_seconds() if job.completed_at else 0,
            "confidence_score": job.confidence_score,
            "total_rows": len(rows),
            "rows_needing_review": sum(1 for row in rows if row["needs_review"])
        }
    }

    json_str = json.dumps(result_data, indent=2, default=str)

    return Response(
        content=json_str,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename={job.job_id}_result.json"
        }
    )


async def _export_csv(job: Job, extraction_results: list, include_provenance: bool):
    """Export results as CSV."""
    output = io.StringIO()

    # Determine all column names
    all_columns = set()
    for result in extraction_results:
        all_columns.update(result.columns_data.keys())

    # Create header
    fieldnames = ["row_id", "page", "region_id", "bbox"] + sorted(all_columns)
    if include_provenance:
        fieldnames.extend([
            "source_file", "source_page", "source_bbox",
            "token_ids", "confidence", "needs_review"
        ])

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    # Write data rows
    for result in extraction_results:
        row_data = {
            "row_id": result.row_id,
            "page": result.page,
            "region_id": result.region_id,
            "bbox": f"[{result.bbox_x1},{result.bbox_y1},{result.bbox_x2},{result.bbox_y2}]"
        }

        # Add column data
        for col in sorted(all_columns):
            row_data[col] = result.columns_data.get(col, "")

        if include_provenance:
            row_data.update({
                "source_file": result.source_file,
                "source_page": result.source_page,
                "source_bbox": f"[{result.source_bbox_x1},{result.source_bbox_y1},{result.source_bbox_x2},{result.source_bbox_y2}]",
                "token_ids": json.dumps(result.token_ids),
                "confidence": result.confidence,
                "needs_review": result.needs_review
            })

        writer.writerow(row_data)

    csv_content = output.getvalue()
    output.close()

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={job.job_id}_result.csv"
        }
    )


async def _export_xlsx(job: Job, extraction_results: list, include_provenance: bool):
    """Export results as Excel file with multiple sheets."""
    output = io.BytesIO()

    # Prepare data for DataFrame
    rows_data = []
    for result in extraction_results:
        row_data = {
            "row_id": result.row_id,
            "page": result.page,
            "region_id": result.region_id,
            "bbox_x1": result.bbox_x1,
            "bbox_y1": result.bbox_y1,
            "bbox_x2": result.bbox_x2,
            "bbox_y2": result.bbox_y2,
            "needs_review": result.needs_review
        }

        # Add column data
        row_data.update(result.columns_data)

        if include_provenance:
            row_data.update({
                "source_file": result.source_file,
                "source_page": result.source_page,
                "source_bbox_x1": result.source_bbox_x1,
                "source_bbox_y1": result.source_bbox_y1,
                "source_bbox_x2": result.source_bbox_x2,
                "source_bbox_y2": result.source_bbox_y2,
                "token_ids": json.dumps(result.token_ids),
                "confidence": result.confidence
            })

        rows_data.append(row_data)

    # Create Excel file with multiple sheets
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main data sheet
        df = pd.DataFrame(rows_data)
        df.to_excel(writer, sheet_name='Extracted_Data', index=False)

        # Summary sheet
        summary_data = {
            "Metric": [
                "Job ID",
                "File Name",
                "Render Type",
                "Total Rows",
                "Rows Needing Review",
                "Processing Time (seconds)",
                "Overall Confidence"
            ],
            "Value": [
                job.job_id,
                job.file_name,
                job.render_type or "Unknown",
                len(extraction_results),
                sum(1 for r in extraction_results if r.needs_review),
                (job.completed_at - job.created_at).total_seconds() if job.completed_at else 0,
                job.confidence_score or 0.0
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Review items sheet (if any)
        review_items = [r for r in extraction_results if r.needs_review]
        if review_items:
            review_data = []
            for item in review_items:
                review_data.append({
                    "row_id": item.row_id,
                    "page": item.page,
                    "confidence": item.confidence,
                    "columns": json.dumps(item.columns_data)
                })
            review_df = pd.DataFrame(review_data)
            review_df.to_excel(writer, sheet_name='Review_Items', index=False)

    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.read()),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={job.job_id}_result.xlsx"
        }
    )


@router.get("/result/{job_id}/provenance")
async def download_provenance_zip(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user)
):
    """
    Download complete provenance package as ZIP file.

    Includes:
    - Original file
    - Extraction results (JSON, CSV, XLSX)
    """
    # This endpoint is intentionally left unimplemented in the minimal run.
    raise HTTPException(status_code=501, detail="Provenance ZIP not implemented in minimal run")


def _results_to_csv_bytes(results_json: dict, include_provenance: bool) -> str:
    """Convert saved JSON results to CSV string."""
    import csv
    from io import StringIO

    output = StringIO()

    # Flatten pages -> regions -> tokens and tables into rows
    writer = None

    def ensure_writer(fieldnames):
        nonlocal writer
        if writer is None:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

    # Basic row list: page, type, text, confidence, bbox, plus provenance
    for page in results_json.get("pages", []):
        page_num = page.get("page_num") or page.get("page") or 0

        for region in page.get("text_regions", []) or page.get("blocks", []) or page.get("regions", []):
            row = {
                "page": page_num,
                "type": region.get("type", "text"),
                "text": region.get("text", ""),
                "confidence": region.get("confidence", 0.0),
                "bbox": json.dumps(region.get("bbox") or region.get("box") or []),
            }
            if include_provenance:
                row.update({
                    "engine": region.get("attributes", {}).get("engine") if isinstance(region.get("attributes"), dict) else None,
                })
            ensure_writer(list(row.keys()))
            writer.writerow(row)

        # Tables
        for table in page.get("tables", []) or []:
            data = table.get("data", [])
            for r_idx, row_cells in enumerate(data):
                for c_idx, cell in enumerate(row_cells):
                    row = {
                        "page": page_num,
                        "type": f"table_cell",
                        "row": r_idx + 1,
                        "col": c_idx + 1,
                        "text": cell,
                    }
                    ensure_writer(list(row.keys()))
                    writer.writerow(row)

    return output.getvalue()


def _results_to_xlsx_bytes(results_json: dict, include_provenance: bool) -> bytes:
    """Convert saved JSON results to XLSX bytes with multiple sheets."""
    import pandas as pd
    from io import BytesIO

    buf = BytesIO()

    # Build DataFrames
    rows = []
    tables_rows = []

    for page in results_json.get("pages", []):
        page_num = page.get("page_num") or page.get("page") or 0

        for region in page.get("text_regions", []) or page.get("blocks", []) or page.get("regions", []):
            rows.append({
                "page": page_num,
                "type": region.get("type", "text"),
                "text": region.get("text", ""),
                "confidence": region.get("confidence", 0.0),
                "bbox": json.dumps(region.get("bbox") or region.get("box") or []),
            })

        for t_idx, table in enumerate(page.get("tables", []) or []):
            data = table.get("data", [])
            for r_idx, row_cells in enumerate(data):
                for c_idx, cell in enumerate(row_cells):
                    tables_rows.append({
                        "page": page_num,
                        "table": t_idx + 1,
                        "row": r_idx + 1,
                        "col": c_idx + 1,
                        "text": cell,
                    })

    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Extracted_Data", index=False)
        if tables_rows:
            pd.DataFrame(tables_rows).to_excel(writer, sheet_name="Tables", index=False)

        # Summary
        meta = results_json.get("summary") or {}
        meta_items = [{"Metric": k, "Value": v} for k, v in meta.items()]
        if meta_items:
            pd.DataFrame(meta_items).to_excel(writer, sheet_name="Summary", index=False)

    return buf.getvalue()

