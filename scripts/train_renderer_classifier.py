#!/usr/bin/env python3
"""
Train renderer classifier using data in data/raw/<slug>.

This script is a thin wrapper around training/train.py to reduce friction.

Usage:
  python scripts/train_renderer_classifier.py --data data/raw/<slug> --output models/checkpoints/renderer_classifier/<slug>
"""
import argparse
from pathlib import Path
import sys

# Reuse training infrastructure
from training.train import main as train_main, load_config, create_model
from training.utils import setup_logging, setup_training
from training.data_loader import create_data_loaders
from training.trainer import CurioScanTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train renderer classifier")
    p.add_argument("--data", required=True, help="Path to dataset root (unzipped)")
    p.add_argument("--output", required=True, help="Output directory for checkpoints")
    p.add_argument("--epochs", type=int, default=1, help="Epochs for quick run")
    p.add_argument("--config", default="configs/demo.yaml", help="Training config YAML")
    return p.parse_args()


def _maybe_write_annotations(data_root: Path) -> None:
    """If <split>_annotations.json missing, create them by scanning subfolders and splitting.
    Expects a folder structure like: data_root/<class_name>/*.jpg
    """
    import json, random
    random.seed(42)

    # Collect samples
    class_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    samples = []
    class_to_idx = {cls.name: i for i, cls in enumerate(sorted(class_dirs))}
    for cls in class_dirs:
        for img in cls.rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                samples.append({
                    "image_path": str(img.relative_to(data_root)).replace("\\", "/"),
                    "label": class_to_idx[cls.name],
                    "metadata": {}
                })

    if not samples:
        return

    # Split 80/10/10
    random.shuffle(samples)
    n = len(samples)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    train = samples[:n_train]
    val = samples[n_train:n_train+n_val]
    test = samples[n_train+n_val:]

    def write(split, items):
        (data_root / f"{split}_annotations.json").write_text(json.dumps(items, indent=2))

    # Only write if not present
    for split, items in [("train", train), ("val", val), ("test", test)]:
        f = data_root / f"{split}_annotations.json"
        if not f.exists():
            write(split, items)


def main():
    args = parse_args()
    setup_logging()
    config = load_config(args.config)

    # Auto-create annotations if needed
    data_root = Path(args.data)
    _maybe_write_annotations(data_root)

    # Patch config paths to point to provided data directory
    config = {
        **config,
        "data": {
            "train_path": str(data_root),
            "val_path": str(data_root),
            "test_path": str(data_root),
            **config.get("data", {}),
        },
        "trainer": {
            **config.get("trainer", {}),
            "max_epochs": args.epochs,
        },
        "model": {
            **config.get("model", {}),
            "type": "renderer_classifier",
        }
    }

    device, world_size, rank = setup_training(config, local_rank=-1)

    model = create_model("renderer_classifier", config).to(device)

    train_loader, val_loader, test_loader = create_data_loaders(config, world_size, rank)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = CurioScanTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=out_dir,
        is_distributed=False,
        rank=rank,
    )

    trainer.train(train_loader=train_loader, val_loader=val_loader, start_epoch=0)

    # Evaluate and save metrics.json
    if test_loader is not None:
        metrics = trainer.evaluate(test_loader)
        (out_dir / "metrics.json").write_text(__import__("json").dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

