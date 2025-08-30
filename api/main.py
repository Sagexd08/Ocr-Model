from fastapi import FastAPI
from .routers import upload, status, review, webhooks, results
from .database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])
app.include_router(results.router, prefix="/api/v1", tags=["results"])
app.include_router(review.router, prefix="/api/v1", tags=["review"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "CurioScan API"}
