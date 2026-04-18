# v4 — Optimised for 45K Combined Dataset (2026-04-18 17:21)

## What changed from v3

v4 keeps the same backbone and dataset as v3 but applies a full set of modern training
techniques to maximise accuracy and prevent overfitting on the combined 45K dataset.

### Changes

| Change | Effect |
|---|---|
| IMG_SIZE 96 → 112 | More detail from higher-res AffectNet/RAFDB images |
| ImageNet normalisation stats | Correct colour calibration for pretrained backbone |
| MixUp (α=0.2) | Blends two training images — biggest anti-overfitting technique |
| Label smoothing (ε=0.1) | Prevents overconfident predictions |
| GaussianBlur + RandomGrayscale aug | Handles noisy internet photos and FER-2013 grayscale origin |
| Cosine LR + linear warmup | Smoother convergence than ReduceLROnPlateau |
| Gradient clipping (norm=1.0) | Stability on mixed-quality dataset |
| EMA weights (decay=0.9995) | Saved model is a running average — more stable than last checkpoint |
| Simplified head: Dropout(0.5) → Linear(1280, 3) | Removes 256-dim bottleneck, less overfitting for 3 classes |

### Training setup

| Setting | v3 | v4 |
|---|---|---|
| Image size | 96×96 | 112×112 |
| Normalisation | Custom RGB stats | ImageNet stats |
| MixUp | — | α=0.2 |
| Label smoothing | — | ε=0.1 |
| LR schedule | ReduceLROnPlateau | Cosine + linear warmup |
| Gradient clipping | — | max_norm=1.0 |
| EMA | — | decay=0.9995 |
| Classifier head | BN → Dropout(0.4) → 256 → ReLU → Dropout(0.3) → 3 | Dropout(0.5) → 3 |
| Epochs | 50 | 90 (15 + 25 + 50, early stopped) |

---

## Results

| Metric | v3 | v4 | Change |
|---|---|---|---|
| **Overall Test Accuracy** | **76.00%** | **77.65%** | **+1.65%** |
| Best val accuracy | — | 78.4% | — |

---

## Files

```
EFFICIENTNET_B0.py                   — Training script (v4)
efficientnet_b0_emotion.pth          — Trained EMA model weights
training_history.json                — Per-epoch metrics
training_curves_efficientnet_b0.png  — Training curves
confusion_matrix_efficientnet_b0.png — Confusion matrix
requirements.txt                     — Python dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
```

## Training

```bash
python EFFICIENTNET_B0.py
```
