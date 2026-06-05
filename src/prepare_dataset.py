"""
Convert PolypDB COCO annotations → YOLO format
Filters by modality (default: WLI only)

Usage:
    python src/prepare_dataset.py --data_root ./data/polypdb --modality WLI
"""

import json
import os
import shutil
import argparse
from pathlib import Path


MODALITY_PREFIXES = {
    "WLI":  "WLI",
    "NBI":  "NBI",
    "BLI":  "BLI",
    "FICE": "FICE",
    "LCI":  "LCI",
}


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] → YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return cx, cy, w_n, h_n


def process_split(coco_json_path: Path, images_src_dir: Path, out_dir: Path, modality: str):
    """Process one split (train/val/test) and write YOLO labels + copy images."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Index annotations by image_id
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    prefix = MODALITY_PREFIXES[modality]
    kept = 0

    for img_info in coco["images"]:
        fname = img_info["file_name"]
        # Filter by modality prefix
        if not os.path.basename(fname).upper().startswith(prefix):
            continue

        img_id = img_info["id"]
        img_w = img_info["width"]
        img_h = img_info["height"]
        stem = Path(fname).stem

        # Write YOLO label file
        label_path = labels_dir / f"{stem}.txt"
        anns = ann_by_img.get(img_id, [])
        with open(label_path, "w") as lf:
            for ann in anns:
                if ann.get("bbox") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
                    cx, cy, w, h = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                    lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Copy image
        src = images_src_dir / fname
        dst = images_dir / os.path.basename(fname)
        if src.exists():
            shutil.copy2(src, dst)
        kept += 1

    print(f"  [{modality}] kept {kept} images from {Path(coco_json_path).name}")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/polypdb", help="Root of downloaded PolypDB")
    parser.add_argument("--out_dir", default="./data/polypdb_wli", help="Output directory for YOLO dataset")
    parser.add_argument("--modality", default="WLI", choices=list(MODALITY_PREFIXES.keys()))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)

    splits = ["train", "val", "test"]
    for split in splits:
        coco_json = data_root / "coco_labels" / f"{split}.json"
        images_src = data_root / "images"
        out_split = out_root / split

        if not coco_json.exists():
            print(f"  WARNING: {coco_json} not found — skipping")
            continue

        print(f"Processing {split}...")
        process_split(coco_json, images_src, out_split, args.modality)

    print(f"\nDataset ready at: {out_root}")
    print("Next: update configs/wli_yolov8n.yaml path, then run train.py")


if __name__ == "__main__":
    main()
