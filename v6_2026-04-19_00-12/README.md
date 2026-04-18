# v6 — Fine-tuned InceptionResNetV1 with CutMix + TTA (2026-04-19 00:12)

## What changed from v5

v6 resumes directly from v5's trained weights (86.68%) and applies targeted
fine-tuning techniques to push accuracy further. No architecture change — same
InceptionResNetV1 backbone pretrained on VGGFace2.

### Changes

| Change | Effect |
|---|---|
| Resume from v5 weights | Starts at 86.68% — no wasted epochs relearning basics |
| CutMix (50/50 with MixUp) | Cuts a facial region from one image and pastes into another — forces model to read the whole face rather than relying on one shortcut feature |
| TTA at validation and test | Averages predictions over original + horizontal flip — free accuracy gain |
| Label smoothing 0.1 → 0.05 | Model is already well-calibrated; less smoothing allows sharper decision boundaries |
| Dropout 0.5 → 0.3 | EMA handles regularisation; less dropout for polishing vs learning from scratch |
| Single phase at LR=5e-6 | Gentle micro-corrections on a converged model — 100× smaller than v5 phase 3 |

### Training

| Setting | v5 | v6 |
|---|---|---|
| Starting point | VGGFace2 pretrained | v5 checkpoint (86.68%) |
| Augmentation | MixUp only | CutMix + MixUp (random 50/50) |
| TTA at inference | No | Yes (original + h-flip average) |
| Label smoothing | 0.1 | 0.05 |
| Dropout | 0.5 | 0.3 |
| LR | 1e-5 (phase 3) | 5e-6 |
| Patience | 12 | 15 |
| Epochs | 66 (3 phases) | 27 (1 phase, early stopped) |

---

## Results

| Version | Backbone | Test Accuracy |
|---|---|---|
| v4 | EfficientNet-B0 (ImageNet) | 77.65% |
| v5 | InceptionResNetV1 (VGGFace2) | 86.68% |
| **v6** | **InceptionResNetV1 (VGGFace2) — fine-tuned** | **87.25%** |

**+0.57% over v5.** Best val accuracy reached 88.6% during training.

---

## Files

```
INCEPTION_RESNET_V1.py              — Training script (v6)
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
