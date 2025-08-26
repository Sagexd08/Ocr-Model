from fastapi import APIRouter

router = APIRouter()

@router.post("/webhooks/register")
async def register_webhook():
    return {"message": "webhook registered"}