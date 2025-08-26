from fastapi import APIRouter, UploadFile, File
from ..worker.tasks import process_document
from ..storage import StorageManager
import uuid

router = APIRouter()
storage_manager = StorageManager()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_name = f"{uuid.uuid4()}-{file.filename}"
    storage_manager.save_file(file_name, file.file)
    task = process_document.delay(file_name)
    return {"job_id": task.id}

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    task = process_document.AsyncResult(job_id)
    response = {
        "status": task.status,
        "result": task.result,
        "job_id": job_id
    }
    return response

@router.get("/result/{job_id}")
async def get_job_result(job_id: str):
    task = process_document.AsyncResult(job_id)
    if task.state == 'SUCCESS':
        return task.result
    else:
        return {"status": task.state}
