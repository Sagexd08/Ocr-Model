"""
FastAPI dependencies for CurioScan API.
"""

import time
import logging
from typing import Optional
from functools import wraps

from fastapi import HTTPException, Depends, UploadFile, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import redis

from .config import get_settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)
settings = get_settings()

# Redis client for rate limiting
redis_client = redis.from_url(settings.redis_url)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[dict]:
    """
    Get current user from JWT token.
    
    Returns None if no authentication is required or token is invalid.
    """
    if not settings.api_key_required:
        return None
    
    if not credentials:
        if settings.api_key_required:
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )
        return None
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
        
        return {
            "user_id": user_id,
            "username": payload.get("username"),
            "email": payload.get("email"),
            "roles": payload.get("roles", [])
        }
        
    except JWTError as e:
        logger.warning(f"JWT decode error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


async def rate_limit(request: Request) -> None:
    """
    Rate limiting dependency.
    
    Limits requests per minute based on client IP.
    """
    if not settings.rate_limit_enabled:
        return
    
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    
    try:
        # Get current request count
        current_requests = redis_client.get(key)
        
        if current_requests is None:
            # First request from this IP
            redis_client.setex(key, 60, 1)  # Set with 60 second expiry
        else:
            current_count = int(current_requests)
            
            if current_count >= settings.rate_limit_requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            # Increment counter
            redis_client.incr(key)
    
    except redis.RedisError as e:
        logger.warning(f"Redis error in rate limiting: {str(e)}")
        # Continue without rate limiting if Redis is unavailable


async def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file.
    
    Checks file size, type, and content.
    """
    # Check file size
    if hasattr(file, 'size') and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
        )
    
    # Check MIME type
    if file.content_type not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Supported types: {', '.join(settings.allowed_file_types)}"
        )
    
    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    # Basic filename validation
    if len(file.filename) > 255:
        raise HTTPException(
            status_code=400,
            detail="Filename too long (max 255 characters)"
        )
    
    # Check for potentially dangerous filenames
    dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
    if any(char in file.filename for char in dangerous_chars):
        raise HTTPException(
            status_code=400,
            detail="Invalid characters in filename"
        )


def require_role(required_role: str):
    """
    Dependency factory for role-based access control.
    
    Usage: @app.get("/admin", dependencies=[Depends(require_role("admin"))])
    """
    def role_checker(current_user: dict = Depends(get_current_user)):
        if not current_user:
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )
        
        user_roles = current_user.get("roles", [])
        if required_role not in user_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{required_role}' required"
            )
        
        return current_user
    
    return role_checker


def log_request_time(func):
    """
    Decorator to log request processing time.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            processing_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {processing_time:.3f}s")
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {processing_time:.3f}s: {str(e)}")
            raise
    
    return wrapper


class MetricsCollector:
    """
    Dependency for collecting API metrics.
    """
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
    
    async def __call__(self, request: Request):
        """Collect request metrics."""
        self.request_count += 1
        request.state.start_time = time.time()
        
        return self
    
    def record_error(self):
        """Record an error."""
        self.error_count += 1
    
    def record_completion(self, processing_time: float):
        """Record successful completion."""
        self.total_processing_time += processing_time
    
    def get_metrics(self) -> dict:
        """Get collected metrics."""
        avg_processing_time = (
            self.total_processing_time / max(1, self.request_count - self.error_count)
        )
        
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(1, self.request_count),
            "average_processing_time": avg_processing_time
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()


async def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return metrics_collector
