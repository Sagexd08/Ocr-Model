"""
CurioScan FastAPI Main Application

This module contains the main FastAPI application with all endpoints for the CurioScan OCR system.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from .config import get_settings
from .database import get_db, init_db
from .models import JobStatus, JobResponse, UploadResponse, StatusResponse, WebhookRequest
from .dependencies import get_current_user, rate_limit
from .routers import upload, status, results, webhooks, review, retrain
from .middleware import PrometheusMiddleware, LoggingMiddleware, metrics_endpoint
from .exceptions import setup_exception_handlers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting CurioScan API...")
    
    # Initialize database
    await init_db()
    
    # Initialize storage
    from .storage import init_storage
    await init_storage()
    
    # Initialize models
    from .ml_service import init_models
    await init_models()
    
    logger.info("CurioScan API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CurioScan API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="CurioScan API",
        description="Production-grade OCR system with intelligent document processing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
    app.include_router(status.router, prefix="/api/v1", tags=["status"])
    app.include_router(results.router, prefix="/api/v1", tags=["results"])
    app.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])
    app.include_router(review.router, prefix="/api/v1", tags=["review"])
    app.include_router(retrain.router, prefix="/api/v1", tags=["retrain"])

    # Add metrics endpoint
    app.get("/metrics")(metrics_endpoint)
    
    return app


# Create the app instance
app = create_app()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "curioscan-api"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "CurioScan API",
        "version": "1.0.0",
        "description": "Production-grade OCR system",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers if not settings.debug else 1
    )
