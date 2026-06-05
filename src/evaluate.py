"""
Evaluate trained model — mAP50, inference speed, model size.
Exports to ONNX and reports file size.

Usage:
    python src/evaluate.py --weights runs/train/yolov8n_wli/weights/best.pt
"""

import argparse
import time
import os
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--data", default="configs/wli_yolov8n.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    # --- Validation metrics ---
    print("Running validation...")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    # --- Inference speed (CPU, single image) ---
    import torch
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    # Warmup
    for _ in range(5):
        model.predict(source=dummy, verbose=False, device="cpu")
    # Time 20 runs
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict(source=dummy, verbose=False, device="cpu")
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = np.mean(times)

    # --- Model size ---
    pt_size_mb = os.path.getsize(args.weights) / 1e6

    # --- ONNX export ---
    print("Exporting to ONNX...")
    onnx_path = Path(args.weights).parent / "best.onnx"
    model.export(format="onnx", imgsz=args.imgsz)
    onnx_size_mb = os.path.getsize(onnx_path) / 1e6 if onnx_path.exists() else None

    # --- Report ---
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"mAP@50:        {map50:.4f}  ({map50*100:.1f}%)")
    print(f"mAP@50-95:     {map50_95:.4f}  ({map50_95*100:.1f}%)")
    print(f"Inference (CPU): {avg_ms:.1f} ms/frame  ({1000/avg_ms:.1f} FPS)")
    print(f"Model size (.pt): {pt_size_mb:.1f} MB")
    if onnx_size_mb:
        print(f"Model size (.onnx): {onnx_size_mb:.1f} MB")
    print("="*50)


if __name__ == "__main__":
    main()
