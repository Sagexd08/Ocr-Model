"""
Configuration management for CurioScan API.
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://curioscan:curioscan123@localhost:5432/curioscan",
        env="DATABASE_URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Storage Configuration
    storage_type: str = Field(default="minio", env="STORAGE_TYPE")  # minio, s3, local
    
    # MinIO Configuration
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin123", env="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")
    minio_bucket: str = Field(default="curioscan", env="MINIO_BUCKET")
    
    # S3 Configuration
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    s3_bucket: str = Field(default="curioscan-prod", env="S3_BUCKET")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    api_key_required: bool = Field(default=False, env="API_KEY_REQUIRED")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )
    
    # File Upload Configuration
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    allowed_file_types: List[str] = Field(
        default=[
            "application/pdf",
            "image/jpeg",
            "image/png", 
            "image/tiff",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Processing Configuration
    default_confidence_threshold: float = Field(default=0.8, env="DEFAULT_CONFIDENCE_THRESHOLD")
    max_processing_time: int = Field(default=600, env="MAX_PROCESSING_TIME")  # 10 minutes
    
    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Monitoring Configuration
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # Model Configuration
    models_path: str = Field(default="models", env="MODELS_PATH")
    device: str = Field(default="auto", env="DEVICE")  # auto, cpu, cuda
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
