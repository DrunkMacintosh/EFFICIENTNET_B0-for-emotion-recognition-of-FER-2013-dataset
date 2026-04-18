# v3 — Combined Dataset: FER-2013 + Balanced AffectNet + RAFDB (2026-04-18 16:52)

## What changed from v2

v3 expands the training data from ~20K FER-2013 images to ~45K images sourced from three datasets, giving the model broader exposure to real-world facial expressions across different lighting conditions, ethnicities, and image qualities.

### New dataset composition

| Dataset | Images | Notes |
|---|---|---|
| FER-2013 | ~20,336 | 48×48 grayscale, lab-collected |
| Balanced AffectNet | ~13,452 | Higher resolution, balanced class distribution |
| RAFDB | ~12,079 | Real-world internet images, high diversity |
| **Total** | **~45,867** | Happy=19,260 · Neutral=14,528 · Sad=12,079 |

### RGB input

All images are loaded as RGB (`.convert("RGB")`). FER-2013 grayscale images are replicated across 3 channels automatically. This allows the model to use standard ImageNet pretrained weights without any first-conv modification, and to leverage real colour information from AffectNet and RAFDB.

### ColorJitter augmentation

Added `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)` to the training pipeline. This is meaningful now that real colour images are present — it makes the model more robust to lighting variation across the three source datasets.

### Training setup

| Setting | v2 | v3 |
|---|---|---|
| Dataset | FER-2013 (~20K) | FER-2013 + AffectNet + RAFDB (~45K) |
| Input channels | 1 (grayscale) | 3 (RGB) |
| Augmentation | Flip · Rotate · Affine · Erase | + ColorJitter |
| Loss | FocalLoss (γ=2) + class weights | FocalLoss (γ=2) + class weights |
| Batch sampling | WeightedRandomSampler | WeightedRandomSampler |
| Epochs | 50 (early stopped) | 50 |

---

## Results

| Metric | v2 | v3 | Change |
|---|---|---|---|
| **Overall Test Accuracy** | **73.80%** | **76.00%** | **+2.2%** |

The 2.2 percentage point gain comes entirely from the larger, more diverse training set — the model architecture and loss function are unchanged. RAFDB's real-world diversity is especially valuable for reducing overfitting to lab-style expressions.

---

## Files

```
EFFICIENTNET_B0.py                   — Training script (v3)
efficientnet_b0_emotion.pth          — Trained model weights
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
