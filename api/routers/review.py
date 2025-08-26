from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session
from .. import models
from ..database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/review/{job_id}")
async def get_review_data(job_id: str, db: Session = Depends(get_db)):
    document = db.query(models.Document).filter(models.Document.job_id == job_id).first()
    return document.content

@router.post("/review/{job_id}")
async def save_review_data(job_id: str, corrected_document: dict = Body(...), db: Session = Depends(get_db)):
    document = db.query(models.Document).filter(models.Document.job_id == job_id).first()
    corrected_data = models.CorrectedData(
        document_id=document.id,
        corrected_content=corrected_document
    )
    db.add(corrected_data)
    db.commit()
    db.refresh(corrected_data)
    return {"message": "Corrected data saved successfully"}