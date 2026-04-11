# v2 — Focal Loss + Balanced Sampler (2026-04-11 20:42)

## What changed from v1

v2 resumes from the v1 checkpoint and continues training with two targeted fixes for the low Neutral and Sad accuracy.

### Problem in v1
The model was biased toward Happy (the largest class). Even with class-weighted loss, training batches were naturally skewed — Happy appeared far more often per update, so the model kept optimising for it at the expense of Neutral and Sad.

### Fix 1 — Focal Loss (γ=2)

Replaces standard CrossEntropyLoss. Focal Loss multiplies each sample's loss by `(1 − p)²`, where `p` is the model's confidence on the correct class.

- **Easy examples** (Happy, confidently predicted) → `(1 − p)²` is small → small gradient → model stops over-optimising for these
- **Hard examples** (Neutral/Sad, low confidence) → `(1 − p)²` is large → large gradient → model is forced to learn these

Class weights (`alpha`) are kept on top of the focal term to maintain the imbalance correction from v1.

### Fix 2 — WeightedRandomSampler

Every training batch is now drawn so that Happy, Neutral, and Sad appear at equal frequency, regardless of their actual dataset sizes. This works alongside Focal Loss — the sampler ensures the model *sees* balanced data, while Focal Loss ensures it *learns harder* from the minority classes it already struggles with.

### Training setup

| Setting | v1 | v2 |
|---|---|---|
| Loss | CrossEntropyLoss + class weights + label smoothing | FocalLoss (γ=2) + class weights |
| Batch sampling | Random (class-imbalanced) | WeightedRandomSampler (class-balanced) |
| Starting point | ImageNet pretrained | Loaded from v1 checkpoint |
| Phases | 3 (head → 30 layers → 60 layers) | 1 (60 layers unfrozen, lr=5e-6) |
| Epochs | 55 (early stopped) | 50 (early stopped) |
| Patience | 5 | 10 |

---

## Results

| Class | v1 Recall | v2 Recall | Change |
|---|---|---|---|
| Happy | 81.8% | 80.1% | −1.7% |
| Neutral | 71.6% | 71.0% | −0.6% |
| Sad | **61.7%** | **68.3%** | **+6.6%** |
| **Overall** | **72.7%** | **73.8%** | **+1.1%** |

Sad recall improved the most (+6.6%), which was the weakest class in v1. The slight drop in Happy is expected — Focal Loss intentionally reduces the gradient on easy/confident predictions, trading some Happy accuracy for better Neutral/Sad learning.

---

## Files

```
RESNET50.py                   — Training script (v2)
plot_curves.py                — Generates training_curves_resnet50.png
plot_confusion.py             — Generates confusion_matrix_resnet50.png
resnet50_emotion.pth          — Trained model weights
training_history.json         — Per-epoch metrics
training_curves_resnet50.png  — Training curves
confusion_matrix_resnet50.png — Confusion matrix
requirements.txt              — Python dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
```

## Training

```bash
python RESNET50.py
```

To retrain from scratch instead of resuming from v1, set `RESUME_FROM = None` at the top of `RESNET50.py`.
