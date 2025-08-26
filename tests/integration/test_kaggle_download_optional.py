import os
import sys
import pytest
from pathlib import Path

KAGGLE_AVAILABLE = False
try:
    import kaggle  # noqa: F401
    KAGGLE_AVAILABLE = True
except Exception:
    KAGGLE_AVAILABLE = False


@pytest.mark.skipif(not KAGGLE_AVAILABLE or not os.getenv("RUN_KAGGLE_TESTS"), reason="Kaggle not available or disabled")
def test_download_small_dataset(tmp_path, monkeypatch):
    # Only run when credentials and flag are present
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        pytest.skip("Missing Kaggle credentials")

    # Use a very small public dataset to keep the test fast
    dataset = os.getenv("KAGGLE_TEST_DATASET", "zynicide/wine-reviews")

    target_dir = tmp_path / "raw"
    cmd = [sys.executable, "scripts/download_kaggle_dataset.py", "--dataset", dataset, "--unzipped-dir", str(target_dir)]

    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Assert basic success
    assert proc.returncode == 0, proc.stderr
    assert target_dir.exists()
    files = [p for p in target_dir.rglob('*') if p.is_file()]
    assert len(files) > 0

