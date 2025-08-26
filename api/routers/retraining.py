from fastapi import APIRouter
from ..worker.retraining_tasks import retrain_model

router = APIRouter()

@router.post("/retrain-trigger")
async def trigger_retraining():
    retrain_model.delay()
    return {"message": "Retraining triggered"}