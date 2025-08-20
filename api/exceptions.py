"""
Exception handlers for CurioScan API.
"""

import logging
from typing import Union

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError
import redis.exceptions

logger = logging.getLogger(__name__)


class CurioScanException(Exception):
    """Base exception for CurioScan API."""
    
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ProcessingException(CurioScanException):
    """Exception raised during document processing."""
    
    def __init__(self, message: str, job_id: str = None, details: dict = None):
        super().__init__(message, status_code=422, details=details)
        self.job_id = job_id


class StorageException(CurioScanException):
    """Exception raised during storage operations."""
    
    def __init__(self, message: str, operation: str = None, path: str = None):
        details = {}
        if operation:
            details["operation"] = operation
        if path:
            details["path"] = path
        
        super().__init__(message, status_code=500, details=details)


class ValidationException(CurioScanException):
    """Exception raised during input validation."""
    
    def __init__(self, message: str, field: str = None, value: str = None):
        details = {}
        if field:
            details["field"] = field
        if value:
            details["value"] = value
        
        super().__init__(message, status_code=400, details=details)


class AuthenticationException(CurioScanException):
    """Exception raised during authentication."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class AuthorizationException(CurioScanException):
    """Exception raised during authorization."""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=403)


class RateLimitException(CurioScanException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


async def curioscan_exception_handler(request: Request, exc: CurioScanException):
    """Handler for CurioScan custom exceptions."""
    logger.error(
        f"CurioScan exception: {exc.message}",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "exception_type": type(exc).__name__,
            "status_code": exc.status_code,
            "details": exc.details
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": type(exc).__name__,
                "message": exc.message,
                "details": exc.details
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )


async def http_exception_handler(request: Request, exc: Union[HTTPException, StarletteHTTPException]):
    """Handler for HTTP exceptions."""
    logger.warning(
        f"HTTP exception: {exc.detail}",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "status_code": exc.status_code,
            "url": str(request.url)
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler for request validation errors."""
    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "url": str(request.url),
            "errors": exc.errors()
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "ValidationError",
                "message": "Request validation failed",
                "details": exc.errors()
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )


async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handler for database exceptions."""
    logger.error(
        f"Database error: {str(exc)}",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "DatabaseError",
                "message": "Database operation failed",
                "details": {"error": str(exc)}
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )


async def redis_exception_handler(request: Request, exc: redis.exceptions.RedisError):
    """Handler for Redis exceptions."""
    logger.error(
        f"Redis error: {str(exc)}",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "exception_type": type(exc).__name__
        }
    )
    
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "RedisError",
                "message": "Cache service unavailable",
                "details": {"error": str(exc)}
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handler for unhandled exceptions."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "exception_type": type(exc).__name__,
            "url": str(request.url)
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc)}
            },
            "request_id": getattr(request.state, "request_id", None)
        }
    )


def setup_exception_handlers(app: FastAPI):
    """Setup all exception handlers for the FastAPI app."""
    
    # Custom exception handlers
    app.add_exception_handler(CurioScanException, curioscan_exception_handler)
    app.add_exception_handler(ProcessingException, curioscan_exception_handler)
    app.add_exception_handler(StorageException, curioscan_exception_handler)
    app.add_exception_handler(ValidationException, curioscan_exception_handler)
    app.add_exception_handler(AuthenticationException, curioscan_exception_handler)
    app.add_exception_handler(AuthorizationException, curioscan_exception_handler)
    app.add_exception_handler(RateLimitException, curioscan_exception_handler)
    
    # Standard exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, database_exception_handler)
    app.add_exception_handler(redis.exceptions.RedisError, redis_exception_handler)
    
    # Catch-all handler
    app.add_exception_handler(Exception, general_exception_handler)
