"""
train.py
--------
Fine-tunes YOLOv8n on the converted parking-space dataset.

Pre-requisites
--------------
    1. Run  python convert_dataset.py  to generate dataset/
    2. pip install ultralytics

Usage
-----
    python train.py
    python train.py --epochs 30 --device cpu
    python train.py --epochs 50 --batch 8 --device 0
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT      = Path(__file__).parent
YAML_PATH = ROOT / "dataset" / "dataset.yaml"
MODELS    = ROOT / "models"


# ── CLI args ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 parking-space detector")
    p.add_argument("--weights",  default="yolov8n.pt", help="Base weights")
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device",   default="",   help="'' = auto | 'cpu' | '0'")
    p.add_argument("--workers",  type=int, default=4)
    return p.parse_args()


# ── Pre-flight checks ───────────────────────────────────────────────────────────

def check_environment():
    if sys.version_info < (3, 9):
        sys.exit("[ERROR] Python 3.9+ required.")

    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed.\n  Run: pip install ultralytics")

    if not YAML_PATH.exists():
        sys.exit(
            f"[ERROR] dataset.yaml not found at:\n  {YAML_PATH}\n\n"
            "Run convert_dataset.py first:\n  python convert_dataset.py"
        )

    train_dir = ROOT / "dataset" / "images" / "train"
    n = len(list(train_dir.glob("*.png"))) if train_dir.exists() else 0
    if n == 0:
        sys.exit("[ERROR] No training images found. Run convert_dataset.py first.")

    val_dir = ROOT / "dataset" / "images" / "val"
    n_val   = len(list(val_dir.glob("*.png"))) if val_dir.exists() else 0
    print(f"Dataset OK — train: {n}  val: {n_val}")


# ── Training ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    check_environment()

    from ultralytics import YOLO

    MODELS.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("  Parking Space Detector — Training")
    print("=" * 50)
    print(f"  Weights  : {args.weights}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Img size : {args.imgsz}")
    print(f"  Patience : {args.patience}")
    print(f"  Device   : {args.device or 'auto'}")
    print(f"  YAML     : {YAML_PATH}\n")

    model = YOLO(args.weights)

    results = model.train(
        data      = str(YAML_PATH),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        patience  = args.patience,
        device    = args.device,
        workers   = args.workers,
        project   = str(MODELS),
        name      = "parking_detector",
        save      = True,
        exist_ok  = True,
        plots     = True,
        verbose   = True,
    )

    # ── Copy best.pt to predictable location ───────────────────────────────────
    best_src = MODELS / "parking_detector" / "weights" / "best.pt"
    best_dst = MODELS / "best.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n✅  Best model saved → {best_dst}")
    else:
        print(f"\n[WARN] best.pt not found at {best_src}")

    # ── Metrics ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Training Metrics")
    print("=" * 50)
    try:
        rd = results.results_dict
        metric_keys = [
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
        ]
        for k in metric_keys:
            if k in rd:
                print(f"  {k:<30s}: {rd[k]:.4f}")
    except Exception:
        print("  (See models/parking_detector/results.csv for full metrics)")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()


