# polyp-detect

YOLOv8n polyp detector on PolypDB — Phase 1 MVP

**Dataset:** [PolypDB](https://osf.io/pr7ms/) — 3,934 endoscopy images, 5 modalities (WLI/NBI/BLI/FICE/LCI)  
**Phase 1 scope:** WLI-only (White Light Imaging) — 90.4% of dataset  
**Architecture:** YOLOv8n (nano) — optimized for edge deployment  
**SOTA reference:** GDCA-Net mAP@50 = 85.9% (target to beat/match)

---

## Results

| Metric | Value |
|---|---|
| mAP@50 | TBD |
| mAP@50-95 | TBD |
| Inference (T4 GPU) | TBD ms |
| Model size (.pt) | TBD MB |
| Model size (.onnx) | TBD MB |

---

## Quickstart

### 1. Prepare dataset
```bash
# Download PolypDB from OSF first → place at data/polypdb/
python src/prepare_dataset.py --data_root ./data/polypdb --modality WLI
```

### 2. Train (local GPU)
```bash
python src/train.py --epochs 100 --batch 16 --device 0
```

### 3. Train (Kaggle T4 — recommended)
Upload `notebooks/polypdb_yolov8n_wli.ipynb` to Kaggle and run all cells.  
Enable GPU: Settings → Accelerator → GPU T4 x2

### 4. Evaluate
```bash
python src/evaluate.py --weights runs/train/yolov8n_wli/weights/best.pt
```

---

## Project Structure
```
polyp-detect/
├── configs/
│   └── wli_yolov8n.yaml     # Dataset config for YOLO
├── notebooks/
│   └── polypdb_yolov8n_wli.ipynb   # Kaggle training notebook
├── src/
│   ├── prepare_dataset.py   # COCO → YOLO conversion, WLI filter
│   ├── train.py             # Training script
│   └── evaluate.py          # mAP + speed + size + ONNX export
└── data/                    # (gitignored) downloaded dataset
```

---

## Dataset Details

| Modality | Images | % of total |
|---|---|---|
| WLI (White Light) | ~3,558 | 90.4% |
| NBI | ~146 | 3.7% |
| BLI | ~60 | 1.5% |
| FICE | ~88 | 2.2% |
| LCI | ~82 | 2.1% |

Phase 2 will add cross-modality evaluation: train on WLI → test on NBI/BLI.

---

## References
- Paper: [PolypDB arxiv 2409.00045](https://arxiv.org/abs/2409.00045)
- Dataset: [OSF osf.io/pr7ms](https://osf.io/pr7ms/)
- GitHub: [DebeshJha/PolypDB](https://github.com/DebeshJha/PolypDB)
- Ultralytics YOLOv8: [docs.ultralytics.com](https://docs.ultralytics.com)
