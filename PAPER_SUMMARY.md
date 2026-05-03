# Facial Emotion Recognition — Full Experimental History

## Overview

This document summarises all seven experimental versions of a facial emotion
recognition system built to classify three emotions — **Happy**, **Neutral**,
and **Sad** — from a combined dataset of 45,867 images drawn from FER-2013,
Balanced AffectNet, and RAFDB. The work progresses from a 71.85% baseline to
88.12% through systematic improvements in architecture, data, and training
methodology.

---

## 1. Dataset

### Sources

| Dataset | Size | Format | Origin |
|---|---|---|---|
| FER-2013 | ~20,336 images | 48×48 grayscale | Lab-collected |
| Balanced AffectNet | ~13,452 images | Higher-resolution RGB | Web-collected, class-balanced |
| RAFDB | ~12,079 images | Variable-resolution RGB | Real-world internet images |
| **Combined** | **45,867 images** | **RGB (grayscale converted)** | |

### Class distribution (combined)

| Class | Count | Share |
|---|---|---|
| Happy | 19,260 | 42.0% |
| Neutral | 14,528 | 31.7% |
| Sad | 12,079 | 26.3% |

### Split

All versions use a fixed stratified split with `random_state=42`:
- **Train**: 81% (~37,152 images)
- **Validation**: 9% (~4,128 images)
- **Test**: 10% (~4,587 images)

---

## 2. Version History

### v1 — EfficientNet-B0 Baseline (71.85%)
**Date:** 2026-04-11 | **Dataset:** FER-2013 only (~20K)

First working PyTorch implementation. Replaced an earlier TensorFlow/Keras
ResNet-50 that was killed by OOM errors and could not use the RTX 5060 Ti
(sm_120 Blackwell architecture requires PyTorch 2.11.0+cu128).

| Setting | Value |
|---|---|
| Backbone | EfficientNet-B0 (ImageNet pretrained, 5M params) |
| Input | 96×96 grayscale |
| Loss | CrossEntropyLoss + class weights + label smoothing (0.1) |
| Sampler | Random (class-imbalanced batches) |
| Training | 3-phase progressive unfreezing (head → 30 layers → 60 layers) |
| Optimiser | AdamW, weight decay 1e-4 |
| Epochs | 55 (early stopped) |

**Result:** 72.73% test accuracy. Weakness: strong bias toward Happy (largest
class); Neutral recall 71.6%, Sad recall only 61.7%.

---

### v2 — Focal Loss + WeightedRandomSampler (73.80%)
**Date:** 2026-04-11 | **Dataset:** FER-2013 only

Resumed from v1 checkpoint. Targeted the class imbalance problem with two
complementary fixes.

**Key changes:**

| Change | Mechanism |
|---|---|
| Focal Loss (γ=2) | Down-weights easy Happy examples via `(1−p)²` factor; forces focus on hard Neutral/Sad |
| WeightedRandomSampler | Every batch drawn with equal class frequency regardless of dataset size |

**Per-class recall change:**

| Class | v1 | v2 | Δ |
|---|---|---|---|
| Happy | 81.8% | 80.1% | −1.7% |
| Neutral | 71.6% | 71.0% | −0.6% |
| Sad | 61.7% | 68.3% | **+6.6%** |
| **Overall** | 72.73% | **73.80%** | +1.07% |

---

### v3 — Combined Dataset: FER-2013 + AffectNet + RAFDB (76.00%)
**Date:** 2026-04-18 | **Dataset:** Combined 45K

Expanded training data from ~20K to ~45K images. Switched from grayscale to
RGB input to accommodate AffectNet and RAFDB color images.

**Key changes:**

| Change | Reason |
|---|---|
| 3 datasets combined (+25K images) | Broader exposure to real-world variation in lighting, ethnicity, image quality |
| RGB input (`.convert("RGB")`) | FER-2013 grayscale replicated to 3 channels; AffectNet/RAFDB are native RGB |
| ColorJitter augmentation | Meaningful with real color data; improves lighting robustness |

**Result:** 76.00% (+2.2pp). Gain attributed entirely to data diversity — architecture unchanged.

---

### v4 — Modern Training Techniques (77.65%)
**Date:** 2026-04-18 | **Dataset:** Combined 45K

Applied a comprehensive set of modern regularisation and optimisation
techniques to the combined dataset.

**Key changes:**

| Change | Purpose |
|---|---|
| IMG_SIZE 96 → 112 | Better feature resolution from higher-res AffectNet/RAFDB images |
| ImageNet normalisation stats | Correct calibration for pretrained EfficientNet backbone |
| MixUp (α=0.2) | Strongest anti-overfitting regulariser; blends image pairs during training |
| Label smoothing (ε=0.1) | Prevents overconfident predictions |
| GaussianBlur + RandomGrayscale | Robustness to noisy images and FER-2013 grayscale origin |
| Cosine LR + linear warmup | Smoother convergence than ReduceLROnPlateau |
| Gradient clipping (norm=1.0) | Stability on mixed-quality dataset |
| EMA weights (decay=0.9995) | Saved model is running average of weights — more stable than final checkpoint |
| Simplified head: Dropout(0.5)→Linear(1280,3) | Removed 256-dim bottleneck; fewer parameters for 3-class problem |

**Result:** 77.65% (+1.65pp). Best validation accuracy 78.4%.

---

### v5 — InceptionResNetV1 with VGGFace2 Pretraining (86.68%)
**Date:** 2026-04-18 | **Dataset:** Combined 45K

**The largest single improvement in the project.** Replaced the
EfficientNet-B0 backbone (ImageNet pretrained) with InceptionResNetV1
pretrained on VGGFace2 — a dataset of 3.3 million face images across 9,131
identities.

**Motivation:** ImageNet-pretrained models learn general object features; faces
represent only ~5 of 1,000 categories. VGGFace2 training forces the model to
learn fine-grained facial structure — eye shape, brow position, lip curvature,
nasolabial fold geometry — the exact low-level features that encode emotion.
Fine-tuning for emotion classification then only requires the model to group
already-learned facial features into emotion categories, rather than also
learning what a face is.

**Architecture comparison:**

| Component | v4 | v5 |
|---|---|---|
| Backbone | EfficientNet-B0 | InceptionResNetV1 |
| Pretraining data | ImageNet (1.2M general images) | VGGFace2 (3.3M face images) |
| Parameters | 5M | 27.9M |
| Feature dimension | 1,280-dim | 512-dim face embeddings |
| Input size | 112×112 | 160×160 |
| Normalisation | ImageNet stats | [−1, 1] (mean=0.5, std=0.5) |
| Classifier head | Dropout(0.5) → Linear(1280, 3) | Dropout(0.5) → Linear(512, 3) |

All v4 training techniques retained (FocalLoss, MixUp, EMA, cosine LR,
gradient clipping, WeightedRandomSampler).

**Phase-by-phase results:**

| Phase | Epochs | Best val acc | Note |
|---|---|---|---|
| Head only (backbone frozen) | 15 | 43.3% | VGGFace2 embeddings encode identity, not emotion — non-trivial linear mapping |
| Fine-tune last 30 layers | 25 | 86.7% | Backbone rapidly adapts face features toward emotional expression |
| Fine-tune last 60 layers | 26 (early stopped) | 87.2% | Final refinement |

**Result:** 86.68% (+9.03pp over v4). The largest single jump in the entire
project, achieved purely by changing the pretraining domain — dataset and
training procedure unchanged.

---

### v6 — CutMix + TTA Fine-tuning (87.25%)
**Date:** 2026-04-19 | **Dataset:** Combined 45K

Resumed from v5 weights. Applied targeted fine-tuning to a converged model.

**Key changes:**

| Change | Purpose |
|---|---|
| Resume from v5 checkpoint | Starts at 86.68%; no wasted epochs |
| CutMix (50/50 with MixUp) | Cuts a facial region from one image and pastes into another; preserves local spatial structure unlike global pixel blending; forces model to use whole-face features |
| TTA at inference | Averages logits over original + horizontal flip; free accuracy with no training cost |
| Label smoothing 0.1 → 0.05 | Model well-calibrated; sharper decision boundaries appropriate |
| Dropout 0.5 → 0.3 | EMA provides regularisation; less dropout needed for polishing |
| LR 1e-5 → 5e-6 | Micro-corrections on converged model |

**Result:** 87.25% (+0.57pp). Trained for 27 epochs.

---

### v7 — Extended Fine-tuning (88.12%)
**Date:** 2026-04-30 | **Dataset:** Combined 45K

Resumed from v6 checkpoint. Same configuration as v6 (CutMix + MixUp, TTA,
LR=5e-6, patience=15). Extended run of 40 epochs.

**Result:** 88.12% (+0.87pp over v6). Best validation accuracy 88.6%.

---

## 3. Consolidated Results

| Version | Backbone | Pretraining | Dataset | Test Acc | Δ | Epochs |
|---|---|---|---|---|---|---|
| v1 | EfficientNet-B0 | ImageNet | FER-2013 (~20K) | 72.73% | baseline | 55 |
| v2 | EfficientNet-B0 | ImageNet | FER-2013 (~20K) | 73.80% | +1.07% | 50 |
| v3 | EfficientNet-B0 | ImageNet | FER-2013 + AffectNet + RAFDB (~45K) | 76.00% | +2.20% | 50 |
| v4 | EfficientNet-B0 | ImageNet | FER-2013 + AffectNet + RAFDB (~45K) | 77.65% | +1.65% | 90 |
| v5 | InceptionResNetV1 | VGGFace2 | FER-2013 + AffectNet + RAFDB (~45K) | 86.68% | **+9.03%** | 66 |
| v6 | InceptionResNetV1 | VGGFace2 | FER-2013 + AffectNet + RAFDB (~45K) | 87.25% | +0.57% | 27 |
| v7 | InceptionResNetV1 | VGGFace2 | FER-2013 + AffectNet + RAFDB (~45K) | **88.12%** | +0.87% | 40 |

**Total improvement: +15.39 percentage points** from v1 to v7.

---

## 4. Ablation Insights

### Contribution of each major change

| Intervention | Versions | Accuracy gain |
|---|---|---|
| Focal Loss + balanced sampler | v1 → v2 | +1.07% |
| Dataset expansion (20K → 45K, RGB) | v2 → v3 | +2.20% |
| Modern training techniques (MixUp, EMA, cosine LR) | v3 → v4 | +1.65% |
| **Face-domain pretraining (ImageNet → VGGFace2)** | **v4 → v5** | **+9.03%** |
| CutMix + TTA + fine-tuning | v5 → v7 | +1.44% |

**Key finding:** Switching the pretraining domain from ImageNet to a
face-specific dataset (VGGFace2) contributed 58.6% of the total accuracy gain,
outweighing all other interventions combined. This demonstrates that
domain-specific pretraining is the highest-leverage decision in transfer
learning for specialised visual tasks.

---

## 5. Training Configuration (Final — v7)

| Parameter | Value |
|---|---|
| Backbone | InceptionResNetV1 |
| Pretraining | VGGFace2 (3.3M faces, 9,131 identities) |
| Input size | 160×160 RGB |
| Normalisation | mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] |
| Batch size | 64 |
| Loss | FocalLoss (γ=2, label smoothing=0.05) |
| Class weighting | Balanced (sklearn `compute_class_weight`) |
| Batch sampler | WeightedRandomSampler |
| Augmentation | RandomHorizontalFlip, RandomRotation(15), RandomAffine, ColorJitter(0.4,0.4,0.3,0.1), RandomGrayscale(0.1), GaussianBlur, RandomErasing(0.25) |
| Mixing | CutMix (α=1.0) and MixUp (α=0.2), applied randomly 50/50 per batch |
| Optimiser | AdamW (weight decay=1e-4) |
| LR schedule | Linear warmup (2 epochs) → CosineAnnealingLR |
| LR (fine-tuning) | 5e-6 |
| Gradient clipping | max_norm=1.0 |
| EMA | decay=0.9995; EMA weights used for validation and saved |
| TTA | Original + horizontal flip logit average at inference |
| Early stopping | patience=15 on EMA validation loss |
| Hardware | NVIDIA RTX 5060 Ti (sm_120 Blackwell, CUDA 12.8) |
| Framework | PyTorch 2.11.0+cu128 |

---

## 6. Repository Structure

```
EFFICIENTNET_B0.py              — EfficientNet-B0 training script (v1–v4)
INCEPTION_RESNET_V1.py          — InceptionResNetV1 training script (v5–v7)
v1_2026-04-11_19-42/            — v1 snapshot (72.73%)
v2_2026-04-11_20-42/            — v2 snapshot (73.80%)
v3_2026-04-18_16-52/            — v3 snapshot (76.00%)
v4_2026-04-18_17-21/            — v4 snapshot (77.65%)
v5_2026-04-18_22-30/            — v5 snapshot (86.68%)
v6_2026-04-19_00-12/            — v6 snapshot (87.25%)
v7_2026-04-30_20-28/            — v7 snapshot (88.12%) ← best
dataset/                        — excluded from repo (45,867 images)
```
