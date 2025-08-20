"""
Storage manager for CurioScan workers.

Handles file operations with the configured storage backend.
"""

import os
import logging
from typing import Optional, BinaryIO

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage operations for workers."""
    
    def __init__(self):
        self.storage_type = os.getenv("STORAGE_TYPE", "minio")
        self._backend = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the storage backend."""
        try:
            if self.storage_type == "minio":
                self._backend = self._create_minio_backend()
            elif self.storage_type == "s3":
                self._backend = self._create_s3_backend()
            elif self.storage_type == "local":
                self._backend = self._create_local_backend()
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_type}")
            
            logger.info(f"Storage backend initialized: {self.storage_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize storage backend: {str(e)}")
            # Fallback to local storage
            self._backend = self._create_local_backend()
            logger.info("Falling back to local storage")
    
    def _create_minio_backend(self):
        """Create MinIO storage backend."""
        from minio import Minio
        
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        bucket_name = os.getenv("MINIO_BUCKET", "curioscan")
        
        # Ensure bucket exists
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
        
        return MinIOStorageBackend(client, bucket_name)
    
    def _create_s3_backend(self):
        """Create S3 storage backend."""
        import boto3
        
        client = boto3.client(
            's3',
            region_name=os.getenv("S3_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        
        bucket_name = os.getenv("S3_BUCKET", "curioscan-prod")
        
        return S3StorageBackend(client, bucket_name)
    
    def _create_local_backend(self):
        """Create local storage backend."""
        base_path = os.getenv("LOCAL_STORAGE_PATH", "data/storage")
        return LocalStorageBackend(base_path)
    
    def download_file(self, path: str) -> bytes:
        """Download file from storage."""
        try:
            return self._backend.download_file(path)
        except Exception as e:
            logger.error(f"Failed to download file {path}: {str(e)}")
            raise
    
    def upload_file(self, content: bytes, path: str) -> str:
        """Upload file to storage."""
        try:
            return self._backend.upload_file(content, path)
        except Exception as e:
            logger.error(f"Failed to upload file {path}: {str(e)}")
            raise
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists in storage."""
        try:
            return self._backend.file_exists(path)
        except Exception as e:
            logger.error(f"Failed to check file existence {path}: {str(e)}")
            return False
    
    def delete_file(self, path: str) -> bool:
        """Delete file from storage."""
        try:
            return self._backend.delete_file(path)
        except Exception as e:
            logger.error(f"Failed to delete file {path}: {str(e)}")
            return False


class MinIOStorageBackend:
    """MinIO storage backend implementation."""
    
    def __init__(self, client, bucket_name: str):
        self.client = client
        self.bucket_name = bucket_name
    
    def download_file(self, path: str) -> bytes:
        """Download file from MinIO."""
        response = self.client.get_object(self.bucket_name, path)
        content = response.read()
        response.close()
        response.release_conn()
        return content
    
    def upload_file(self, content: bytes, path: str) -> str:
        """Upload file to MinIO."""
        from io import BytesIO
        
        self.client.put_object(
            self.bucket_name,
            path,
            BytesIO(content),
            length=len(content)
        )
        return path
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists in MinIO."""
        try:
            self.client.stat_object(self.bucket_name, path)
            return True
        except:
            return False
    
    def delete_file(self, path: str) -> bool:
        """Delete file from MinIO."""
        try:
            self.client.remove_object(self.bucket_name, path)
            return True
        except:
            return False


class S3StorageBackend:
    """S3 storage backend implementation."""
    
    def __init__(self, client, bucket_name: str):
        self.client = client
        self.bucket_name = bucket_name
    
    def download_file(self, path: str) -> bytes:
        """Download file from S3."""
        from io import BytesIO
        
        buffer = BytesIO()
        self.client.download_fileobj(self.bucket_name, path, buffer)
        buffer.seek(0)
        return buffer.read()
    
    def upload_file(self, content: bytes, path: str) -> str:
        """Upload file to S3."""
        from io import BytesIO
        
        self.client.upload_fileobj(
            BytesIO(content),
            self.bucket_name,
            path
        )
        return path
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=path)
            return True
        except:
            return False
    
    def delete_file(self, path: str) -> bool:
        """Delete file from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=path)
            return True
        except:
            return False


class LocalStorageBackend:
    """Local filesystem storage backend implementation."""
    
    def __init__(self, base_path: str):
        from pathlib import Path
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, path: str) -> bytes:
        """Read file from local filesystem."""
        file_path = self.base_path / path
        with open(file_path, 'rb') as f:
            return f.read()
    
    def upload_file(self, content: bytes, path: str) -> str:
        """Write file to local filesystem."""
        file_path = self.base_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return str(file_path)
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists locally."""
        file_path = self.base_path / path
        return file_path.exists()
    
    def delete_file(self, path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            file_path = self.base_path / path
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except:
            return False
