from .celery_app import celery_app
from ..api.database import SessionLocal
from ..api import models
from ..training.train import train

@celery_app.task
def retrain_model():
    db = SessionLocal()
    corrected_data = db.query(models.CorrectedData).all()
    db.close()

    print(corrected_data)

