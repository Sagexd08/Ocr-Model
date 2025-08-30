# Local storage for development; no MinIO needed.
# Existing MinIO-based StorageManager intentionally removed for local-first setup.

# Minimal local storage helper for uploads (used by API upload route)
try:
    from pathlib import Path
    async def upload_file_to_storage(content: bytes, relative_path: str) -> str:
        base = Path(__file__).resolve().parent.parent / "data" / "storage"
        full = base / relative_path
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "wb") as f:
            f.write(content)
        return str(full)
except Exception:
    # Fallback sync version if asyncio isn't needed
    import os
    def upload_file_to_storage(content: bytes, relative_path: str) -> str:
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "storage")
        full = os.path.join(base, relative_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(content)
        return full
