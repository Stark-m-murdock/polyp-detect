"""
Train YOLOv8n on PolypDB WLI split.
Run on Kaggle (T4 GPU) or locally.

Usage:
    python src/train.py --epochs 100 --imgsz 640 --batch 16
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained weights")
    parser.add_argument("--data", default="configs/wli_yolov8n.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="yolov8n_wli")
    parser.add_argument("--device", default="0", help="GPU id or 'cpu'")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=20,            # early stopping
        save=True,
        plots=True,
        val=True,
    )

    print("\n=== Training complete ===")
    print(f"Best weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
