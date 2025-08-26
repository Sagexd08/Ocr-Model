from fastapi import FastAPI
from .routers import jobs, review, webhooks, retraining
from .database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(jobs.router)
app.include_router(review.router)
app.include_router(webhooks.router)
app.include_router(retraining.router)

@app.get("/")
async def root():
    return {"message": "CurioScan API"}
