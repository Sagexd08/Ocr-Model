from minio import Minio

class StorageManager:
    def __init__(self, endpoint="minio:9000", access_key="minioadmin", secret_key="minioadmin"):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

    def save_file(self, file_name, file_data, bucket_name="curioscan"):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
        
        self.client.put_object(
            bucket_name,
            file_name,
            file_data,
            length=-1, # Required for streams
            part_size=10*1024*1024
        )

    def get_file(self, file_name, bucket_name="curioscan"):
        return self.client.get_object(bucket_name, file_name)