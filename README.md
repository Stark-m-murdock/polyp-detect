# Polyp Detection — Cross-Modal Generalization Study

**PolypDB · YOLO26s · Trained on WLI → Tested on 4 Unseen Modalities**

> Submitted to BDI National AI Hackathon 2026 — AI for Healthcare track  
> Results: June 12, 2026

---

## The Question This Study Answers

PolypDB contains 5 imaging modalities: WLI, NBI, BLI, FICE, LCI.  
WLI has 3,558 images. Every other modality has 60–146.

Training a separate model per modality overfits the small sets.  
The clinically relevant question is different:

> **Does a model trained only on standard white-light imaging generalize to enhanced imaging modalities it has never seen?**

This study answers that directly — with real error analysis, not just headline metrics.

---

## Results

### Training Performance (WLI, best epoch)

| Metric | Value |
|---|---|
| **mAP@50** | **95.12%** (epoch 96) |
| mAP@50-95 | 78.5% |

### Evaluation — WLI Val Set (538 images)

| Metric | Value |
|---|---|
| Precision | 93.9% |
| Recall | 89.4% |
| F1 | 91.6% |
| mIoU | 84.5% |
| mAP@50 | **90.2%** |
| mAP@50-95 | 75.1% |

### Cross-Modal Generalization (zero-shot — model never saw these modalities)

| Modality | Images | mAP@50 | mAP@50-95 | F1 | Notes |
|---|---|---|---|---|---|
| **LCI** | 60 | **91.5%** | 77.7% | 94.4% | Best generalization — color profile closest to WLI |
| **FICE** | 70 | 77.5% | 68.5% | 86.6% | Partial color overlap with WLI |
| **NBI** | 146 | 67.3% | 50.0% | 75.6% | Vascular enhancement = significant domain shift |
| **BLI** | 70 | 64.7% | 52.2% | 75.5% | Narrow-band blue light = highest domain shift |

### Error Analysis (WLI Val, 538 images)

| Error type | Count | Rate |
|---|---|---|
| Complete miss (0 predictions) | 7 | **1.3%** |
| Low confidence (< 0.5) | 27 | 5.0% |
| Imprecise boxes (IoU 0.5–0.75) | 36 | 6.7% |

**7/538 complete misses** — model is reliable on training modality with very low false-negative rate.

---

## Why the Cross-Modal Gap Exists

**LCI generalizes best** — LCI (Linked Color Imaging) enhances mucosal contrast while preserving white-light color balance. Its color profile overlaps substantially with WLI.

**NBI and BLI drop significantly** — Narrow-band imaging highlights vascular patterns using blue (415nm) and green (540nm) light. This produces fundamentally different color statistics vs WLI. The gap is expected and clinically meaningful: these modalities require either multi-modal training or domain adaptation.

**FICE sits in between** — Flexible spectral imaging color enhancement uses post-processing to simulate narrow-band contrast. Partial WLI overlap explains the middle-ground performance.

---

## Model & Training Config

| Parameter | Value |
|---|---|
| Model | YOLO26s (Ultralytics 8.4.6x) |
| Parameters | 9.47M |
| GFLOPs | 20.5 |
| Epochs | 100 (best weights: epoch 96) |
| Batch size | 32 |
| Image size | 640px |
| Hardware | Kaggle T4 GPU (~2 hours) |
| Training modality | WLI only (`WLI_ONLY = True`) |

**Augmentation:**
```
HSV: h=0.015, s=0.7, v=0.4
Flips: flipud=0.5, fliplr=0.5
mixup=0.1, copy_paste=0.1, close_mosaic=10
```

**Dataset split (WLI):** 2,511 train / 538 val / 539 test (70/15/15)

---

## Dataset

**PolypDB** — Jha et al. 2024 ([arXiv:2409.00045](https://arxiv.org/abs/2409.00045))  
3,904 colonoscopy images · 5 modalities · 3 hospitals (Norway, Sweden, Vietnam)

| Modality | Full name | Total images |
|---|---|---|
| WLI | White Light Imaging | 3,608 |
| NBI | Narrow Band Imaging | 146 |
| BLI | Blue Light Imaging | 70 |
| FICE | Flexible Spectral Imaging Color Enhancement | 70 |
| LCI | Linked Color Imaging | 60 |

---

## Reproduce

```bash
# Install
pip install ultralytics==8.4.6

# Train (WLI only)
python train.py --data polypdb_wli.yaml --model yolo26s.pt --epochs 100 --batch 32 --imgsz 640

# Evaluate cross-modality
python eval.py --weights runs/train/best.pt --data polypdb_nbi.yaml
python eval.py --weights runs/train/best.pt --data polypdb_bli.yaml
python eval.py --weights runs/train/best.pt --data polypdb_fice.yaml
python eval.py --weights runs/train/best.pt --data polypdb_lci.yaml
```

Full notebook on Kaggle: [link coming after BDI results]

---

## What's Next

- [ ] Train on ALL 5 modalities (`WLI_ONLY = False`) — measure cross-modal gap reduction
- [ ] Add YOLOv11s and Faster R-CNN comparison (pending collaborator feedback)
- [ ] Explore domain adaptation techniques for NBI/BLI gap
- [ ] Package as reproducible Kaggle notebook

---

## Related Projects

- [รู้รอบกรุง](https://github.com/Mad-m-dock/ruurobnkrung) — Thai civic RAG assistant (LINE OA)
- [Jenna OS](https://github.com/Mad-m-dock/jenna-os-skeleton) — Autonomous multi-agent AI system

---

*Questions or collaboration: stangykung19@gmail.com*
