#!/usr/bin/env python3
"""
Download a KaggleHub package locally and report its path.

Usage:
  python scripts/download_kagglehub_package.py --package <owner/slug/versions/N> [--target data/raw/<name>] [--copy]

Notes:
  - Uses kagglehub (no Kaggle API key required for public packages)
  - If --copy is provided with --target, contents will be copied to target dir
"""
import argparse
from pathlib import Path
import shutil
import sys

try:
    import kagglehub
except Exception:
    kagglehub = None


def parse_args():
    p = argparse.ArgumentParser(description="Download KaggleHub package")
    p.add_argument("--package", required=True, help="KaggleHub package spec, e.g., 'tatamikenn/text-rendering-ocr-exploit-lb-0-305/versions/2'")
    p.add_argument("--target", default=None, help="Optional target directory to copy contents")
    p.add_argument("--copy", action="store_true", help="Copy contents to target directory")
    return p.parse_args()


def main():
    if kagglehub is None:
        print("ERROR: kagglehub not installed. Install with 'pip install kagglehub'", file=sys.stderr)
        sys.exit(2)

    args = parse_args()

    pkg_path = kagglehub.package_import(args.package)
    print(f"KaggleHub package path: {pkg_path}")

    if args.copy and args.target:
        src = Path(pkg_path)
        dst = Path(args.target)
        dst.mkdir(parents=True, exist_ok=True)
        # Copy tree (skip if same)
        if src.resolve() != dst.resolve():
            # shutil.copytree requires dst not exist; copy manually
            for p in src.rglob('*'):
                rel = p.relative_to(src)
                out = dst / rel
                if p.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, out)
        print(f"Copied contents to: {dst}")


if __name__ == "__main__":
    main()

