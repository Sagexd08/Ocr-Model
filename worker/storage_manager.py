# MinIO removed for local development
# from minio import Minio
from pathlib import Path
from typing import Optional


class StorageManager:
    def __init__(self):
        # Pure local mode
        self.client = None

        # Local storage base
        self.base_dir = Path("./data/storage")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"
        self.temp_dir = self.base_dir / "temp"
        for d in [self.base_dir, self.input_dir, self.output_dir, self.temp_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_temp_dir(self) -> str:
        return str(self.temp_dir)

    def get_result_path(self, job_id: str) -> str:
        return str(self.output_dir / job_id)

    # Local-only mode: skip MinIO save_file
    def save_file(self, file_name, file_data, bucket_name="curioscan"):
        raise NotImplementedError("MinIO save_file disabled in local mode")

    def get_file(self, file_name, bucket_name="curioscan"):
        return self.client.get_object(bucket_name, file_name)
    # Local-only file save: write to input directory
    def save_file(self, file_name, file_data, bucket_name: str = "curioscan"):
        target = self.input_dir / file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        data_bytes = file_data.read() if hasattr(file_data, "read") else file_data
        with open(target, "wb") as f:
            f.write(data_bytes)
        return str(target)

    # Local-only file get: open from input directory/base
    def get_file(self, file_name, bucket_name: str = "curioscan"):
        target = self.input_dir / file_name
        if not target.exists():
            target = self.base_dir / file_name
        return open(target, "rb")
