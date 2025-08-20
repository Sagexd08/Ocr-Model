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
from ..database import get_db, Job, ExtractionResult as DBExtractionResult
from ..models import JobStatus
from ..dependencies import get_current_user
from ..storage import get_file_from_storage

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


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
    
    Supported formats:
    - json: Complete results with provenance
    - csv: Tabular data export
    - xlsx: Excel format with multiple sheets
    """
    try:
        # Find the job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed (status: {job.status})"
            )
        
        # Get extraction results
        extraction_results = job.extraction_results
        
        if not extraction_results:
            raise HTTPException(
                status_code=404,
                detail=f"No results found for job {job_id}"
            )
        
        # Convert to the requested format
        if format.lower() == "json":
            return await _export_json(job, extraction_results, include_provenance)
        elif format.lower() == "csv":
            return await _export_csv(job, extraction_results, include_provenance)
        elif format.lower() == "xlsx":
            return await _export_xlsx(job, extraction_results, include_provenance)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Supported formats: json, csv, xlsx"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download results for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download results: {str(e)}"
        )


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
    - Processing metadata
    - OCR confidence maps
    """
    try:
        # Find the job
        job = db.query(Job).filter(Job.job_id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Job {job_id} is not completed"
            )
        
        # TODO: Implement ZIP creation with all provenance data
        # This would include:
        # - Original file
        # - All export formats
        # - Processing logs
        # - Confidence visualizations
        
        raise HTTPException(
            status_code=501,
            detail="Provenance ZIP download not yet implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create provenance ZIP for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create provenance ZIP: {str(e)}"
        )
