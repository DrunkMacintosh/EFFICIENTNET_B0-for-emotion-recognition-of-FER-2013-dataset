# v7 — Continued Fine-tuning from v6 (2026-04-30 20:28)

## What changed from v6

v7 resumes from v6's trained weights (87.25%) and runs another round of the same
fine-tuning: CutMix + MixUp, TTA, LR=5e-6, patience=15.

Same script as v6 with `RESUME_FROM` pointing to the v6 checkpoint.

### Training

| Setting | Value |
|---|---|
| Starting point | v6 checkpoint (87.25%) |
| LR | 5e-6 |
| Epochs | 22 (early stopped) |
| Best val acc | 88.3% |
| Augmentation | CutMix + MixUp (50/50) |
| TTA | Yes (original + h-flip) |

---

## Results

| Version | Test Accuracy |
|---|---|
| v5 | 86.68% |
| v6 | 87.25% |
| **v7** | **87.77%** |

---

## Files

```
INCEPTION_RESNET_V1.py              — Training script (v7)
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
