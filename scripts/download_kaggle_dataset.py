#!/usr/bin/env python3
"""
Download Kaggle dataset or competition to data/raw/...

Usage:
  python scripts/download_kaggle_dataset.py --dataset <owner/slug> [--unzipped-dir data/raw/<slug>] [--type dataset|competition]

Credentials:
  - Reads KAGGLE_USERNAME and KAGGLE_KEY from environment, or uses ~/.kaggle/kaggle.json

Notes:
  - Keeps everything offline (stored locally). No credentials are written to the repo.
"""
import argparse
import os
from pathlib import Path
import sys

try:
    from kaggle import api as kaggle_api
except Exception as e:
    kaggle_api = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Kaggle dataset or competition")
    p.add_argument("--dataset", required=True, help="Kaggle owner/slug, e.g., 'zynicide/wine-reviews'")
    p.add_argument("--unzipped-dir", default=None, help="Target directory to unzip into (default: data/raw/<slug>)")
    p.add_argument("--type", choices=["dataset", "competition"], default="dataset", help="Resource type to download")
    p.add_argument("--force", action="store_true", help="Force re-download and overwrite")
    return p.parse_args()


def ensure_api():
    if kaggle_api is None:
        print("ERROR: kaggle package not installed. Please install with 'pip install kaggle'", file=sys.stderr)
        sys.exit(2)
    # Authenticate (reads env vars or kaggle.json)
    kaggle_api.authenticate()


def main():
    args = parse_args()
    ensure_api()

    slug = args.dataset.split("/")[-1]
    target_dir = Path(args.unzipped_dir) if args.unzipped_dir else Path("data/raw") / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir.parent / f"{slug}.zip"

    if args.type == "dataset":
        kaggle_api.dataset_download_files(args.dataset, path=str(target_dir.parent), force=args.force, quiet=False)
    else:
        kaggle_api.competition_download_files(args.dataset, path=str(target_dir.parent), force=args.force, quiet=False)

    # Find the downloaded zip
    # Kaggle uses <slug>.zip in dataset mode
    if not zip_path.exists():
        # Fallback: search for any zip dropped in parent
        zips = list(target_dir.parent.glob("*.zip"))
        if not zips:
            print("ERROR: No zip file downloaded.", file=sys.stderr)
            sys.exit(3)
        zip_path = zips[0]

    # Unzip
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)

    # Verify files exist
    files = [p for p in target_dir.rglob('*') if p.is_file()]
    print(f"Downloaded and extracted {len(files)} files to {target_dir}")


if __name__ == "__main__":
    main()

