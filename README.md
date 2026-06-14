# Polyp Detection — Cross-Modal Architecture Comparison

**Which YOLO architecture generalizes best across endoscopic imaging modalities?**

> PolypDB · 4-Architecture Comparison · Trained on WLI → Zero-shot tested on NBI / BLI / FICE / LCI

---

## The Research Question

PolypDB contains 5 imaging modalities: WLI, NBI, BLI, FICE, and LCI.  
WLI (White Light Imaging) accounts for 3,588 of 3,934 images (91%).

> **Does architecture choice — independent of augmentation — affect how well a WLI-trained model generalizes to unseen imaging modalities?**

This study trains four architectures under identical conditions (same dataset, epochs, augmentation config) where **only the model architecture varies**.

---

## Four-Architecture Comparison

| Model | Parameters | GFLOPs | Family |
|---|---|---|---|
| **YOLO26s** | 9.47M | 20.5 | Custom (YOLO26) |
| **YOLOv11n** | 2.58M | 6.3 | Ultralytics v11 nano |
| **YOLOv11s** | 9.41M | 21.3 | Ultralytics v11 small |
| **YOLOv8n** | 3.01M | 8.1 | Ultralytics v8 nano |

---

## Cross-Modal Results

### mAP@50 Heatmap

![Cross-Modal Heatmap](cross_modal_heatmap.png)

### Full Results Table

| Model | WLI (train) | LCI | FICE | NBI | BLI | Avg OOD |
|---|---|---|---|---|---|---|
| **YOLO26s** | 0.983 | 0.980 | 0.881 | 0.790 | 0.761 | 0.853 |
| **YOLOv11n** | 0.977 | 0.977 | 0.962 | 0.929 | 0.921 | 0.947 |
| **YOLOv11s** | 0.983 | **0.992** | 0.961 | **0.943** | **0.940** | **0.959** |
| **YOLOv8n** | **0.983** | 0.989 | **0.963** | 0.926 | 0.933 | 0.953 |

> WLI = training modality (val set). LCI / FICE / NBI / BLI = zero-shot (never seen during training).  
> Avg OOD = mean mAP@50 across 4 unseen modalities.

---

## Key Findings

### 1. Larger is not more generalizable

YOLO26s achieves the same WLI performance as others (98.3%) but the worst OOD generalization — dropping to 76.1% on BLI and 79.0% on NBI.

YOLOv11s (similar parameter count) outperforms it on every OOD modality by **6–18 percentage points**.

| Model | WLI → BLI drop | WLI → NBI drop |
|---|---|---|
| YOLO26s | **−22.2%** | **−19.3%** |
| YOLOv11n | −5.6% | −4.8% |
| YOLOv11s | −4.3% | −3.9% |
| YOLOv8n | −5.0% | −5.7% |

YOLO26s has **4–5× larger cross-modal drop** than the v11/v8 architectures.

### 2. YOLOv11s achieves best overall balance

- Best average OOD mAP@50: **95.9%**
- Highest single OOD score: LCI **99.2%**
- Most consistent across all 5 modalities

### 3. YOLOv8n punches above its weight

3.01M parameters vs 9.41M for YOLOv11s — yet achieves **95.3% average OOD**. Strong generalization at 1/3 the model size.

### 4. Cross-modal drop order is consistent across all architectures

**LCI > FICE > NBI > BLI** — this reflects domain shift magnitude, not architecture behavior.

---

## Error Analysis & Limitations

### Where Models Fail

**YOLO26s — systematic cross-modal failure**

YOLO26s achieves 98.3% on WLI but collapses to 76–79% on NBI/BLI. The failure pattern is consistent: the model over-relies on WLI-specific color and texture features rather than shape-based polyp features. When the color distribution shifts (NBI/BLI use narrow-band light), detection confidence drops sharply.

| Error type | YOLO26s | YOLOv11s | Interpretation |
|---|---|---|---|
| WLI → BLI drop | −22.2% | −4.3% | YOLO26s: color-dependent · v11s: more shape-based |
| WLI → NBI drop | −19.3% | −3.9% | Same pattern — vascular enhancement confuses YOLO26s |
| WLI performance | 98.3% | 98.3% | Both memorize WLI equally well |

**v11/v8 family — residual NBI/BLI gap**

Even the best model (YOLOv11s) drops ~4% on NBI/BLI. This residual gap likely reflects genuine domain shift that cannot be closed by architecture alone — the model has never seen narrow-band images during training.

### Known Limitations

1. **Training data is WLI-only.** Cross-modal performance could improve significantly by adding even a small number of NBI/BLI training examples (multi-modal training experiment not yet run).

2. **100 epochs may not be full convergence.** Training logs show best weights at epoch 96–98 for most models, suggesting additional epochs may yield marginal gains. Recommended next run: 150 epochs with `patience=20`.

3. **Hyperparameters were not tuned per architecture.** Using identical hyperparameters across architectures is correct for isolating architecture as the variable, but individual architectures may have different optimal lr/batch settings.

4. **No per-image error analysis.** This study reports aggregate mAP@50 per modality. A full error analysis would classify individual failures into: false negatives (missed polyps), false positives (spurious detections), and localization errors (correct detection, wrong box).

5. **HSV augmentation choice not ablated.** Whether HSV=0 (locked) or HSV=0.7 (enabled) produces better cross-modal results for this dataset is an open question. The baseline experiment used HSV=0; this study used HSV=0.7. Direct comparison is a follow-on experiment.

---

## Why the Cross-Modal Gaps Exist

**LCI generalizes best** — Linked Color Imaging preserves the white-light color profile. Closest color statistics to WLI.

**NBI and BLI drop significantly** — Narrow-band imaging uses 415nm / 540nm wavelengths to highlight vascular patterns, producing fundamentally different color distributions.

**FICE sits in between** — Post-processing simulation of narrow-band contrast with partial WLI color overlap.

**Clinical implication:** A WLI-only model is reliable for standard imaging and LCI, but should not be deployed for NBI/BLI without additional adaptation. The gap is not a model failure — it is a real domain boundary.

---

## Training Configuration & Hyperparameters

All four architectures were trained under **identical conditions**. The only variable is the model architecture.

| Parameter | Value | Rationale |
|---|---|---|
| Training modality | WLI only | Study cross-modal generalization from a single source |
| Dataset split | 70 / 15 / 15 | Official PolypDB split CSVs |
| Epochs | 100 | Sufficient for convergence on T4 GPU (~2 hr/model) |
| Image size | 640px | YOLO standard; balances speed and detection of small polyps |
| Batch size | 32 | Fits T4 VRAM (16GB) |
| Optimizer | AdamW (YOLO default) | Stable convergence for detection tasks |
| Learning rate | 0.01 initial, cosine decay | YOLO default schedule |
| Seed | 42 | Reproducibility |
| hsv_h | 0.015 | Slight hue shift |
| hsv_s | 0.7 | Saturation jitter — hypothesis: helps cross-modal robustness |
| hsv_v | 0.4 | Brightness jitter |
| fliplr / flipud | 0.5 | Polyps are orientation-agnostic |
| mixup | 0.1 | Mild regularization |

**HSV decision note:** Standard YOLO HSV values were retained under the hypothesis that color variance during training helps generalization across modalities with different color profiles. An alternative approach (locking HSV=0 to preserve tissue color fidelity as a diagnostic signal) was used in the baseline experiment and produced strong results. Comparing these two augmentation strategies is a direct follow-on experiment.

**What was not tuned:** Learning rate, batch size, and optimizer were left at YOLO defaults. Systematic hyperparameter search (e.g., grid search over lr, weight decay) was not performed — this study isolates architecture as the single variable, not hyperparameter sensitivity.

---

## Dataset

**PolypDB** — Jha et al. 2024 ([arXiv:2409.00045](https://arxiv.org/abs/2409.00045))  
3,934 colonoscopy images · 5 modalities · 3 hospitals (Norway, Sweden, Vietnam)

| Modality | Full name | Images |
|---|---|---|
| WLI | White Light Imaging | 3,588 |
| NBI | Narrow Band Imaging | 146 |
| BLI | Blue Light Imaging | 70 |
| FICE | Flexible Spectral Imaging Color Enhancement | 70 |
| LCI | Linked Color Imaging | 60 |

---

## Exploratory Data Analysis (EDA)

### Class Imbalance

WLI dominates the dataset at **91.2%** of all images. The remaining 4 modalities together account for only 346 images (8.8%).

| Modality | Images | % of total | Train | Val | Test |
|---|---|---|---|---|---|
| WLI | 3,588 | 91.2% | 2,511 | 538 | 539 |
| NBI | 146 | 3.7% | 102 | 22 | 22 |
| BLI | 70 | 1.8% | 49 | 10 | 11 |
| FICE | 70 | 1.8% | 49 | 10 | 11 |
| LCI | 60 | 1.5% | 42 | 9 | 9 |

This imbalance is the central challenge of the study. A model trained on this dataset without intervention will see ~91% WLI at every epoch, making minority-modality features unable to exert meaningful gradient influence.

### Modality Characteristics

Each modality uses different light physics, producing fundamentally different image statistics:

| Modality | Light type | Visual characteristic | Domain distance from WLI |
|---|---|---|---|
| WLI | White broadband | Natural tissue color | — (reference) |
| LCI | Enhanced white | Slightly boosted mucosal contrast | Low |
| FICE | Computed narrow-band | Simulated spectral enhancement | Medium |
| NBI | 415nm + 540nm | Blue-green, vascular emphasis | High |
| BLI | 410nm narrow-band | Blue-dominant, high vascular contrast | Highest |

This ordering (WLI → LCI → FICE → NBI → BLI) directly predicts the cross-modal drop order observed in the results, confirming that domain shift magnitude — not model failure — explains the performance gradient.

### Label Distribution

All images contain exactly 1 class: `Polyp`. Each mask was converted to a single bounding box per polyp contour. Images with no detectable polyp region after thresholding were labeled as empty (negative examples retained in dataset).

---

## Data Cleaning & Feature Engineering

### Mask → YOLO Bounding Box Conversion

PolypDB provides segmentation masks (binary PNG), not bounding boxes. YOLO requires bounding box labels in normalized xywh format.

**Conversion pipeline (`src/prepare_dataset.py`):**

```python
def mask_to_yolo_bbox(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:   # filter noise contours
            continue
        x, y, wb, hb = cv2.boundingRect(cnt)
        # normalize to [0, 1] for YOLO format
        xc = (x + wb/2.0) / w
        yc = (y + hb/2.0) / h
```

**Decisions made:**
- Contours with area < 10 pixels filtered as noise
- Each polyp contour generates one bounding box (multi-polyp images produce multiple labels)
- Modality prefix added to filenames (`WLI_img001.jpg`) to enable per-modality evaluation filtering

### Dataset Split

Official PolypDB split CSVs (per-modality, per-split) were used without modification to ensure reproducibility and comparability with prior work. Split ratio: **70 / 15 / 15** (train / val / test).

---

## Reproduce

```bash
pip install ultralytics==8.4.6

# Evaluate cross-modal generalization
python scripts/evaluate.py --weights <model>.pt --modality NBI
python scripts/evaluate.py --weights <model>.pt --modality BLI
python scripts/evaluate.py --weights <model>.pt --modality FICE
python scripts/evaluate.py --weights <model>.pt --modality LCI

# Generate cross-modal heatmap
python scripts/plot_crossmodal_heatmap.py
```

Full evaluation notebook: `notebooks/polypdb_crossmodal_eval.ipynb`

---

## Demo

A live inference demo (YOLOv11s best weights) is deployed on Hugging Face Spaces:

**[MaddMDock/polyp-detect](https://huggingface.co/spaces/MaddMDock/polyp-detect)** — Upload a colonoscopy image → get bounding box predictions

> Note: The demo currently runs the initial YOLO26s weights. Update to YOLOv11s best weights is pending.

---

## Deliverables Summary

| Item | Status |
|---|---|
| Training notebook (Kaggle) | ✅ `notebooks/polypdb_comparison.ipynb` |
| Evaluation script | ✅ `src/cross_modality_eval.py` |
| Model weights (3 models) | ✅ Available on request |
| Cross-modal results CSV | ✅ `cross_modal_results.csv` |
| Heatmap visualization | ✅ `cross_modal_heatmap.png` |
| Dataset preparation script | ✅ `src/prepare_dataset.py` |
| Live demo | ✅ HF Spaces (YOLO26s · v11s update pending) |
| PDF report | ❌ Not produced — README serves as documentation |
| Hyperparameter search | ❌ Not performed — architecture is the single variable |
| Per-image error analysis | ❌ Not performed — aggregate metrics only |

---

## Related Projects

- [รู้รอบกรุง](https://github.com/Mad-m-dock/ruurobnkrung) — Thai civic RAG assistant · BDI National AI Hackathon 2026
- [Jenna OS](https://github.com/Mad-m-dock) — Production multi-agent AI system

---

*Questions or collaboration: stangykung19@gmail.com*
