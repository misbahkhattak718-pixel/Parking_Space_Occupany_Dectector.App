"""
convert_dataset.py
------------------
Parses annotations.xml (CVAT polygon format), converts polygon points to
axis-aligned bounding-box YOLO labels, splits 80/20 into train/val, and
writes dataset/dataset.yaml.

Label mapping
  free_parking_space           -> 0  (free)
  not_free_parking_space       -> 1  (occupied)
  partially_free_parking_space -> 2  (partially_occupied)

Usage
-----
    python convert_dataset.py
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
ANNOTATIONS = ROOT / "annotations.xml"
SRC_IMAGES  = ROOT / "images"
DATASET_DIR = ROOT / "dataset"

TRAIN_IMG   = DATASET_DIR / "images" / "train"
VAL_IMG     = DATASET_DIR / "images" / "val"
TRAIN_LBL   = DATASET_DIR / "labels" / "train"
VAL_LBL     = DATASET_DIR / "labels" / "val"
YAML_PATH   = DATASET_DIR / "dataset.yaml"

# ── Class mapping ──────────────────────────────────────────────────────────────
LABEL_MAP = {
    "free_parking_space":            0,
    "not_free_parking_space":        1,
    "partially_free_parking_space":  2,
}
CLASS_NAMES = ["free", "occupied", "partially_occupied"]

SPLIT_RATIO = 0.8
RANDOM_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def polygon_to_bbox(points_str: str):
    """
    Convert CVAT polygon string "x1,y1;x2,y2;..." to axis-aligned
    (xmin, ymin, xmax, ymax).  Returns None on bad input.
    """
    try:
        coords = [
            (float(p.split(",")[0]), float(p.split(",")[1]))
            for p in points_str.strip().split(";")
            if "," in p
        ]
    except ValueError:
        return None

    if len(coords) < 2:
        return None

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert absolute pixel bbox to YOLO normalised (cx, cy, w, h)."""
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin)        / img_w
    bh = (ymax - ymin)        / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return cx, cy, bw, bh


# ── Parse ──────────────────────────────────────────────────────────────────────

def parse_annotations(xml_path: Path) -> dict:
    """
    Returns
    -------
    dict keyed by image stem, e.g. "0":
        {
          'src_name': 'images/0.png',
          'width':    1200,
          'height':   621,
          'labels':   [(class_id, cx, cy, bw, bh), ...],
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data: dict = {}
    skipped: set = set()

    for img_elem in root.iter("image"):
        src_name = img_elem.get("name", "")
        img_w    = max(int(img_elem.get("width",  1)), 1)
        img_h    = max(int(img_elem.get("height", 1)), 1)
        stem     = Path(src_name).stem

        labels = []
        for poly in img_elem.findall("polygon"):
            raw_label  = poly.get("label", "").strip()
            points_str = poly.get("points", "")

            if raw_label not in LABEL_MAP:
                skipped.add(raw_label)
                continue

            bbox = polygon_to_bbox(points_str)
            if bbox is None:
                continue

            xmin, ymin, xmax, ymax = bbox
            if xmax <= xmin or ymax <= ymin:
                continue

            cx, cy, bw, bh = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            labels.append((LABEL_MAP[raw_label], cx, cy, bw, bh))

        data[stem] = {
            "src_name": src_name,
            "width":    img_w,
            "height":   img_h,
            "labels":   labels,
        }

    if skipped:
        print(f"  [WARN] Unknown labels skipped: {skipped}")

    return data


# ── Write ──────────────────────────────────────────────────────────────────────

def write_label(path: Path, labels: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
             for cid, cx, cy, bw, bh in labels]
    path.write_text("\n".join(lines) + "\n")


def write_yaml(path: Path):
    cfg = {
        "path":  str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    3,
        "names": CLASS_NAMES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Parking Space Dataset Converter")
    print("=" * 50)

    # Create output dirs
    for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nParsing {ANNOTATIONS.name} ...")
    records = parse_annotations(ANNOTATIONS)
    print(f"  Annotated images found : {len(records)}")

    # Keep only records with an image file on disk
    valid = []
    for stem, info in records.items():
        img_path = SRC_IMAGES / f"{stem}.png"
        if not img_path.exists():
            print(f"  [SKIP] Image not on disk: {img_path.name}")
            continue
        if not info["labels"]:
            print(f"  [SKIP] No valid labels for: {stem}")
            continue
        valid.append(stem)

    print(f"  Valid (image + labels)  : {len(valid)}\n")

    if not valid:
        print("[ERROR] No valid samples found. Check your dataset paths.")
        return

    # Shuffle & split
    random.seed(RANDOM_SEED)
    random.shuffle(valid)
    cut        = int(len(valid) * SPLIT_RATIO)
    train_set  = valid[:cut]
    val_set    = valid[cut:]
    print(f"  Train : {len(train_set)}")
    print(f"  Val   : {len(val_set)}\n")

    # Copy images + write label files
    total_boxes = 0
    for split, stems, img_dir, lbl_dir in [
        ("train", train_set, TRAIN_IMG, TRAIN_LBL),
        ("val",   val_set,   VAL_IMG,   VAL_LBL),
    ]:
        for stem in stems:
            info = records[stem]
            shutil.copy2(SRC_IMAGES / f"{stem}.png", img_dir / f"{stem}.png")
            write_label(lbl_dir / f"{stem}.txt", info["labels"])
            total_boxes += len(info["labels"])

    write_yaml(YAML_PATH)

    # ── Report ─────────────────────────────────────────────────────────────────
    counts = {0: 0, 1: 0, 2: 0}
    for lbl_dir in [TRAIN_LBL, VAL_LBL]:
        for txt in lbl_dir.glob("*.txt"):
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1

    print("=" * 50)
    print("  Conversion Complete")
    print("=" * 50)
    print(f"  Train images  : {len(train_set)}")
    print(f"  Val   images  : {len(val_set)}")
    print(f"  Total boxes   : {total_boxes}")
    print(f"  dataset.yaml  : {YAML_PATH}")
    print("\n  Class distribution:")
    for cid, cname in enumerate(CLASS_NAMES):
        print(f"    [{cid}] {cname:<25s}: {counts[cid]}")


if __name__ == "__main__":
    main()
