"""
Custom middleware for CurioScan API.
"""

import time
import logging
import json
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'curioscan_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'curioscan_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

PROCESSING_JOBS = Counter(
    'curioscan_processing_jobs_total',
    'Total number of processing jobs',
    ['status']
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent"),
                "content_type": request.headers.get("content-type")
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "processing_time": processing_time,
                    "response_size": response.headers.get("content-length")
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log error
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "processing_time": processing_time
                },
                exc_info=True
            )
            
            raise


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics collection."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics collection for metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            processing_time = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=self._get_endpoint_label(request),
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=self._get_endpoint_label(request)
            ).observe(processing_time)
            
            return response
            
        except Exception as e:
            # Record error metrics
            processing_time = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=self._get_endpoint_label(request),
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=self._get_endpoint_label(request)
            ).observe(processing_time)
            
            raise
    
    def _get_endpoint_label(self, request: Request) -> str:
        """Get normalized endpoint label for metrics."""
        path = request.url.path
        
        # Normalize paths with IDs
        if "/jobs/" in path:
            return "/jobs/{job_id}"
        elif "/status/" in path:
            return "/status/{job_id}"
        elif "/result/" in path:
            return "/result/{job_id}"
        elif "/review/" in path:
            return "/review/{job_id}"
        elif "/webhooks/" in path and path != "/webhooks/register":
            return "/webhooks/{webhook_id}"
        
        return path


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with additional security headers."""
    
    def __init__(self, app, allow_origins=None, allow_methods=None, allow_headers=None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        origin = request.headers.get("origin")
        if origin and (origin in self.allow_origins or "*" in self.allow_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
        
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Expose-Headers"] = "X-Request-ID"
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = 60  # 1 minute
        self.request_times = {}  # client_ip -> list of request times
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.request_times:
            self.request_times[client_ip] = [
                req_time for req_time in self.request_times[client_ip]
                if current_time - req_time < self.window_size
            ]
        else:
            self.request_times[client_ip] = []
        
        # Check rate limit
        if len(self.request_times[client_ip]) >= self.requests_per_minute:
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json"
            )
        
        # Add current request
        self.request_times[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - len(self.request_times[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_size))
        
        return response


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


def record_job_metric(status: str):
    """Record a processing job metric."""
    PROCESSING_JOBS.labels(status=status).inc()
