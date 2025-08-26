from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import time
import random
import json
from datetime import datetime

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a directory to store uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "storage", "input")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for job data
jobs = {}

@app.get("/")
async def root():
    return {"message": "CurioScan Demo API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Save the file
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}-{file.filename}")
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    # Create a job entry
    jobs[job_id] = {
        "status": "STARTED",
        "file_name": file.filename,
        "file_path": file_path,
        "created_at": datetime.now().isoformat(),
        "result": None
    }
    
    # Start "processing" in the background (simulated)
    # In a real app, this would be a Celery task
    # Here we'll just set a timer to simulate processing
    def process_job():
        time.sleep(3)  # Simulate processing time
        jobs[job_id]["status"] = "SUCCESS"
        
        # Create demo result data with structured content for CSV/Excel export
        jobs[job_id]["result"] = {
            "pages": [
                {
                    "page_number": 1,
                    "tokens": [
                        {"text": "This is a sample document for the CurioScan Demo.", "bbox": [50, 50, 450, 70], "confidence": 0.98, "line": 1, "position": 1},
                        {"text": "It demonstrates how the OCR system would process documents.", "bbox": [50, 85, 450, 105], "confidence": 0.97, "line": 2, "position": 1},
                        {"text": "In a real implementation, this would use actual OCR models.", "bbox": [50, 120, 450, 140], "confidence": 0.96, "line": 3, "position": 1},
                        {"text": "Each paragraph and line is carefully preserved.", "bbox": [50, 155, 450, 175], "confidence": 0.97, "line": 4, "position": 1},
                        {"text": "The system maintains original document layout and spacing.", "bbox": [50, 190, 450, 210], "confidence": 0.95, "line": 5, "position": 1}
                    ],
                    "tables": [
                        {
                            "image": b"Sample table image data",  # In a real app, this would be actual image data
                            "data": [
                                ["Header 1", "Header 2", "Header 3"],
                                ["Row 1, Cell 1", "Row 1, Cell 2", "Row 1, Cell 3"],
                                ["Row 2, Cell 1", "Row 2, Cell 2", "Row 2, Cell 3"]
                            ],
                            "title": "Sample Table 1"
                        }
                    ],
                    "extracted_text": "This is a sample document for the CurioScan Demo.\n\nIt demonstrates how the OCR system would process documents.\n\nIn a real implementation, this would use actual OCR models.\n\nEach paragraph and line is carefully preserved.\n\nThe system maintains original document layout and spacing.",
                    "original_layout": True
                },
                {
                    "page_number": 2,
                    "tokens": [
                        {"text": "This is page 2 of the sample document.", "bbox": [50, 50, 450, 70], "confidence": 0.98, "line": 1, "position": 1},
                        {"text": "It contains some additional text and a table.", "bbox": [50, 85, 450, 105], "confidence": 0.97, "line": 2, "position": 1},
                        {"text": "The document structure is", "bbox": [50, 120, 250, 140], "confidence": 0.99, "line": 3, "position": 1},
                        {"text": "maintained exactly as in the", "bbox": [255, 120, 450, 140], "confidence": 0.98, "line": 3, "position": 2},
                        {"text": "original file.", "bbox": [50, 155, 150, 175], "confidence": 0.97, "line": 4, "position": 1}
                    ],
                    "tables": [
                        {
                            "image": b"Sample table image data",
                            "data": [
                                ["Name", "Age", "Occupation"],
                                ["John Doe", "35", "Engineer"],
                                ["Jane Smith", "28", "Designer"],
                                ["Robert Brown", "42", "Manager"]
                            ],
                            "title": "Sample Table 2"
                        }
                    ],
                    "extracted_text": "This is page 2 of the sample document.\n\nIt contains some additional text and a table.\n\nThe document structure is maintained exactly as in the\noriginal file.",
                    "original_layout": True
                }
            ],
            "document_summary": "This is a 2-page sample document for the CurioScan Demo.",
            "metadata": {
                "file_name": jobs[job_id]["file_name"],
                "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "page_count": 2,
                "table_count": 2
            }
        }
    
    # Start the "processing" in the background
    import threading
    thread = threading.Thread(target=process_job)
    thread.start()
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "status": jobs[job_id]["status"],
        "job_id": job_id
    }

@app.get("/review/{job_id}")
async def get_job_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]["status"] != "SUCCESS":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    
    return jobs[job_id]["result"]

@app.get("/download/{job_id}")
async def download_results(job_id: str, format: str = "csv"):
    """
    Download the extracted data in CSV or Excel format
    """
    from fastapi.responses import StreamingResponse
    import pandas as pd
    import io
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]["status"] != "SUCCESS":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    
    result = jobs[job_id]["result"]
    
    # Create DataFrame for extracted text with preserved formatting
    text_data = []
    for page in result["pages"]:
        # Preserve exact formatting from document
        extracted_text = page.get("extracted_text", "")
        
        # Create a more detailed record with positional information if available
        if page.get("tokens"):
            # Group tokens by lines for accurate layout
            lines = {}
            for token in page["tokens"]:
                line_num = token.get("line", 1)
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append(token)
            
            # Sort tokens within each line
            for line_num in lines:
                lines[line_num] = sorted(lines[line_num], key=lambda t: t.get("position", 0))
            
            # Format with proper positioning 
            for line_num in sorted(lines.keys()):
                line_tokens = lines[line_num]
                for token in line_tokens:
                    text_data.append({
                        "page_number": page["page_number"],
                        "line_number": line_num,
                        "position": token.get("position", 0),
                        "text": token["text"],
                        "confidence": token.get("confidence", 1.0),
                        "x1": token.get("bbox", [0, 0, 0, 0])[0],
                        "y1": token.get("bbox", [0, 0, 0, 0])[1]
                    })
        else:
            # Fallback to simple text if no token information
            text_data.append({
                "page_number": page["page_number"],
                "line_number": 1,
                "position": 1,
                "text": extracted_text,
                "confidence": 1.0,
                "x1": 0,
                "y1": 0
            })
    
    # Create dataframe with proper sorting to maintain document layout
    text_df = pd.DataFrame(text_data)
    if not text_df.empty:
        text_df = text_df.sort_values(by=["page_number", "y1", "x1"])
    
    # Create DataFrame for tables
    tables_data = []
    for page in result["pages"]:
        if page.get("tables"):
            for i, table in enumerate(page["tables"]):
                table_title = table.get("title", f"Table {i+1} (Page {page['page_number']})")
                tables_data.append({
                    "page_number": page["page_number"],
                    "table_number": i+1,
                    "table_title": table_title,
                    "table_data": pd.DataFrame(table["data"])
                })
    
    # Prepare the output file
    output = io.BytesIO()
    
    # Excel format has multiple sheets
    if format.lower() == "excel":
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            text_df.to_excel(writer, sheet_name='Extracted Text', index=False)
            
            # Add tables to separate sheets
            for i, table_info in enumerate(tables_data):
                sheet_name = f"Table_{i+1}"
                table_info["table_data"].to_excel(writer, sheet_name=sheet_name, index=False)
        
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"curiocr_results_{job_id}.xlsx"
    
    # CSV format combines all data
    else:  # default to csv
        # Just save the text for simplicity in CSV format
        text_df.to_csv(output, index=False)
        
        media_type = "text/csv"
        filename = f"curiocr_results_{job_id}.csv"
    
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
