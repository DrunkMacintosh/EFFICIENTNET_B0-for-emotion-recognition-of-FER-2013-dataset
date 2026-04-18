# v5 — InceptionResNetV1 (VGGFace2 Face-Pretrained Backbone) (2026-04-18 22:30)

## What changed from v4

v4 and all previous versions used **EfficientNet-B0 pretrained on ImageNet** — a model that learned to classify 1,000 general object categories (cars, dogs, furniture, etc.), with faces making up only a small fraction of its training data.

v5 replaces the backbone entirely with **InceptionResNetV1 pretrained on VGGFace2** — a model trained exclusively on 3.3 million face images across 9,131 identities. The task it was trained on (face identity verification: "are these two photos the same person?") forces the model to learn the precise facial features that also encode emotion: eye shape, brow position, lip curvature, nasolabial fold depth, and subtle muscle geometry around the mouth and eyes.

This is the reason for the large accuracy jump. The model did not need to first discover what a face is — it arrived already expert in faces, and only needed to learn to group those features into Happy, Neutral, and Sad.

---

## Architecture

| Component | v4 (EfficientNet-B0) | v5 (InceptionResNetV1) |
|---|---|---|
| Backbone | EfficientNet-B0 | InceptionResNetV1 |
| Pretraining | ImageNet (1.2M general images) | VGGFace2 (3.3M face images) |
| Backbone params | 4M | 27.9M |
| Feature output | 1,280-dim | 512-dim face embeddings |
| Input size | 112×112 | 160×160 |
| Normalisation | ImageNet stats | [-1, 1] (mean=0.5, std=0.5) |
| Classifier head | Dropout(0.5) → Linear(1280, 3) | Dropout(0.5) → Linear(512, 3) |

### Why the head is simpler with a larger backbone

The 512-dim output of InceptionResNetV1 is already a highly compressed, face-specific representation — it was trained to encode an entire identity into those 512 numbers. Mapping that directly to 3 emotion classes is a straightforward linear problem. EfficientNet's 1,280-dim output is a general-purpose feature vector that needed more transformation to become emotion-relevant.

---

## Training setup

All v4 training techniques are retained unchanged:

| Technique | Purpose |
|---|---|
| FocalLoss (γ=2) + class weights | Handles Happy/Neutral/Sad imbalance |
| Label smoothing (ε=0.1) | Prevents overconfident predictions |
| WeightedRandomSampler | Balanced batches regardless of class size |
| MixUp (α=0.2) | Strongest single anti-overfitting regulariser |
| EMA weights (decay=0.9995) | Saved model is a running average — more stable |
| Cosine LR + linear warmup | Smooth convergence per phase |
| Gradient clipping (norm=1.0) | Stability on mixed-quality dataset |

### Phase results

| Phase | Epochs | Best val acc |
|---|---|---|
| Head only (backbone frozen) | 15 | 43.3% |
| Fine-tune last 30 layers | 25 | 86.7% |
| Fine-tune last 60 layers | 26 (early stopped) | 87.2% |

Phase 1 accuracy (43.3%) appears low compared to EfficientNet's Phase 1 because the frozen VGGFace2 embeddings encode *identity*, not *emotion* — the head had to learn a non-trivial linear mapping. Once fine-tuning began in Phase 2, the backbone rapidly adapted its face representations toward emotional expression, jumping to 86.7%.

---

## Results

| Version | Backbone | Dataset | Test Accuracy |
|---|---|---|---|
| v1 | EfficientNet-B0 (ImageNet) | FER-2013 | 71.85% |
| v2 | EfficientNet-B0 (ImageNet) | FER-2013 | 73.80% |
| v3 | EfficientNet-B0 (ImageNet) | FER-2013 + AffectNet + RAFDB | 76.00% |
| v4 | EfficientNet-B0 (ImageNet) | FER-2013 + AffectNet + RAFDB | 77.65% |
| **v5** | **InceptionResNetV1 (VGGFace2)** | **FER-2013 + AffectNet + RAFDB** | **86.68%** |

**+9.03 percentage points over v4.** The entire gain comes from swapping the pretraining domain — same dataset, same training techniques, different backbone.

---

## Files

```
INCEPTION_RESNET_V1.py              — Training script (v5)
face_emotion.pth                    — Trained EMA model weights
face_training_history.json          — Per-epoch metrics
training_curves_face_emotion.png    — Training curves
confusion_matrix_face_emotion.png   — Confusion matrix
requirements.txt                    — Python dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
pip install facenet-pytorch --no-deps
pip install tqdm
```

## Training

```bash
python INCEPTION_RESNET_V1.py
```
