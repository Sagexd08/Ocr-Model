"""
Storage abstraction layer for CurioScan API.

Supports multiple storage backends: MinIO, S3, and local filesystem.
"""

import os
import logging
from typing import Optional, BinaryIO
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from minio import Minio
from minio.error import S3Error

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StorageBackend(ABC):
    """Abstract storage backend interface."""
    
    @abstractmethod
    async def upload_file(self, content: bytes, path: str) -> str:
        """Upload file content to storage."""
        pass
    
    @abstractmethod
    async def download_file(self, path: str) -> bytes:
        """Download file content from storage."""
        pass
    
    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if file exists in storage."""
        pass
    
    @abstractmethod
    async def get_file_url(self, path: str, expires_in: int = 3600) -> str:
        """Get a presigned URL for file access."""
        pass


class MinIOBackend(StorageBackend):
    """MinIO storage backend."""
    
    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self.bucket_name = settings.minio_bucket
        
        # Ensure bucket exists
        asyncio.create_task(self._ensure_bucket_exists())
    
    async def _ensure_bucket_exists(self):
        """Ensure the bucket exists."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Failed to create MinIO bucket: {e}")
    
    async def upload_file(self, content: bytes, path: str) -> str:
        """Upload file to MinIO."""
        try:
            from io import BytesIO
            
            # Upload file
            self.client.put_object(
                self.bucket_name,
                path,
                BytesIO(content),
                length=len(content)
            )
            
            logger.info(f"Uploaded file to MinIO: {path}")
            return path
            
        except S3Error as e:
            logger.error(f"Failed to upload file to MinIO: {e}")
            raise
    
    async def download_file(self, path: str) -> bytes:
        """Download file from MinIO."""
        try:
            response = self.client.get_object(self.bucket_name, path)
            content = response.read()
            response.close()
            response.release_conn()
            
            return content
            
        except S3Error as e:
            logger.error(f"Failed to download file from MinIO: {e}")
            raise
    
    async def delete_file(self, path: str) -> bool:
        """Delete file from MinIO."""
        try:
            self.client.remove_object(self.bucket_name, path)
            logger.info(f"Deleted file from MinIO: {path}")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to delete file from MinIO: {e}")
            return False
    
    async def file_exists(self, path: str) -> bool:
        """Check if file exists in MinIO."""
        try:
            self.client.stat_object(self.bucket_name, path)
            return True
        except S3Error:
            return False
    
    async def get_file_url(self, path: str, expires_in: int = 3600) -> str:
        """Get presigned URL for MinIO object."""
        try:
            from datetime import timedelta
            
            url = self.client.presigned_get_object(
                self.bucket_name,
                path,
                expires=timedelta(seconds=expires_in)
            )
            return url
            
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise


class S3Backend(StorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(self):
        self.client = boto3.client(
            's3',
            region_name=settings.s3_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        self.bucket_name = settings.s3_bucket
    
    async def upload_file(self, content: bytes, path: str) -> str:
        """Upload file to S3."""
        try:
            from io import BytesIO
            
            self.client.upload_fileobj(
                BytesIO(content),
                self.bucket_name,
                path
            )
            
            logger.info(f"Uploaded file to S3: {path}")
            return path
            
        except ClientError as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise
    
    async def download_file(self, path: str) -> bytes:
        """Download file from S3."""
        try:
            from io import BytesIO
            
            buffer = BytesIO()
            self.client.download_fileobj(self.bucket_name, path, buffer)
            buffer.seek(0)
            
            return buffer.read()
            
        except ClientError as e:
            logger.error(f"Failed to download file from S3: {e}")
            raise
    
    async def delete_file(self, path: str) -> bool:
        """Delete file from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=path)
            logger.info(f"Deleted file from S3: {path}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete file from S3: {e}")
            return False
    
    async def file_exists(self, path: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=path)
            return True
        except ClientError:
            return False
    
    async def get_file_url(self, path: str, expires_in: int = 3600) -> str:
        """Get presigned URL for S3 object."""
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': path},
                ExpiresIn=expires_in
            )
            return url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise


class LocalBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self):
        self.base_path = Path(settings.local_base_path if hasattr(settings, 'local_base_path') else "data/storage")
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(self, content: bytes, path: str) -> str:
        """Save file to local filesystem."""
        try:
            file_path = self.base_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Saved file locally: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save file locally: {e}")
            raise
    
    async def download_file(self, path: str) -> bytes:
        """Read file from local filesystem."""
        try:
            file_path = self.base_path / path
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to read local file: {e}")
            raise
    
    async def delete_file(self, path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            file_path = self.base_path / path
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted local file: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete local file: {e}")
            return False
    
    async def file_exists(self, path: str) -> bool:
        """Check if file exists locally."""
        file_path = self.base_path / path
        return file_path.exists()
    
    async def get_file_url(self, path: str, expires_in: int = 3600) -> str:
        """Get local file URL (file:// scheme)."""
        file_path = self.base_path / path
        return f"file://{file_path.absolute()}"


# Global storage backend instance
_storage_backend: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """Get the configured storage backend."""
    global _storage_backend
    
    if _storage_backend is None:
        if settings.storage_type == "minio":
            _storage_backend = MinIOBackend()
        elif settings.storage_type == "s3":
            _storage_backend = S3Backend()
        elif settings.storage_type == "local":
            _storage_backend = LocalBackend()
        else:
            raise ValueError(f"Unsupported storage type: {settings.storage_type}")
    
    return _storage_backend


async def init_storage():
    """Initialize storage backend."""
    backend = get_storage_backend()
    logger.info(f"Initialized storage backend: {type(backend).__name__}")


# Convenience functions
async def upload_file_to_storage(content: bytes, path: str) -> str:
    """Upload file to configured storage backend."""
    backend = get_storage_backend()
    return await backend.upload_file(content, path)


async def get_file_from_storage(path: str) -> bytes:
    """Download file from configured storage backend."""
    backend = get_storage_backend()
    return await backend.download_file(path)


async def delete_file_from_storage(path: str) -> bool:
    """Delete file from configured storage backend."""
    backend = get_storage_backend()
    return await backend.delete_file(path)


async def file_exists_in_storage(path: str) -> bool:
    """Check if file exists in configured storage backend."""
    backend = get_storage_backend()
    return await backend.file_exists(path)


async def get_file_url_from_storage(path: str, expires_in: int = 3600) -> str:
    """Get file URL from configured storage backend."""
    backend = get_storage_backend()
    return await backend.get_file_url(path, expires_in)
