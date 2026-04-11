# EfficientNet-B0 / EfficientNet-B0 — Emotion Recognition on FER-2013

Emotion classifier trained on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset (Happy / Neutral / Sad).  
The training script is named `EFFICIENTNET_B0.py` but the current backbone is **EfficientNet-B0** — see [Model Version](#model-version) below.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **71.85%** (previous EfficientNet-B0 baseline) → target 82–85% after full training |
| Classes | Happy, Neutral, Sad |
| Test set size | 2,046 images (10% stratified split) |

Confusion matrix and training curves are generated automatically at the end of each run.

---

## Model Version

### v2 — EfficientNet-B0 (current)

**Key improvements over the original EfficientNet-B0 script:**

| Change | Why |
|---|---|
| EfficientNet-B0 backbone (5 M params) | 23 M → 5 M params — less overfitting on 20 K images |
| 96×96 grayscale input | Was 224×224 fake-RGB — 4.67× upscale caused blur |
| First conv averaged to 1 channel | Preserves ImageNet knowledge; no wasted capacity on 3 identical channels |
| Class-weighted loss + label smoothing 0.1 | Happy is over-represented (8177 vs ~6100); smoothing reduces overconfidence |
| RandomRotation + RandomAffine + RandomErasing | Face-specific augmentation — tilt, shift, partial occlusion |
| AdamW optimiser | Weight decay reduces overfitting |
| 3-phase training (head → last 30 → last 60 layers) | Gradual backbone unfreezing; safer than fine-tuning all at once |
| Stratified train/val/test split | Each split preserves class ratio |

### v1 — EfficientNet-B0 (baseline)
Original script using EfficientNet-B0 with 224×224 fake-RGB input. Reached ~72% test accuracy.  
Preserved in `efficientnet_b0_emotion.py` for reference.

---

## Project Structure

```
EFFICIENTNET_B0.py                  — Main training script (EfficientNet-B0)
plot_curves.py               — Standalone training curve plotter (reads training_history.json)
plot_confusion.py            — Standalone confusion matrix plotter (loads .pth model)
efficientnet_b0_emotion.pth         — Saved model weights (EfficientNet-B0)
training_history.json        — Per-epoch metrics from the latest training run
training_curves_efficientnet_b0.png — Training curves plot (auto-generated)
confusion_matrix_efficientnet_b0.png— Confusion matrix plot (auto-generated)
efficientnet_b0_emotion.py          — Legacy TF/Keras EfficientNet-B0 script (reference)
improved_emotion_model.py    — EfficientNetV2S TF/Keras experiment
optimized_emotion_model.py   — Earlier TF optimisation attempt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
```

> Requires an NVIDIA GPU with CUDA 12.8+ (tested on RTX 5060 Ti).  
> CPU fallback works but is significantly slower.

## Dataset

Download FER-2013 and arrange as:
```
dataset/
  Happy/    *.jpg
  Neutral/  *.jpg
  Sad/      *.jpg
```
The dataset folder is excluded from this repo (`.gitignore`).

## Training

```bash
source .venv/bin/activate
python EFFICIENTNET_B0.py
```

Outputs saved automatically after training:
- `efficientnet_b0_emotion.pth`
- `training_history.json`
- `training_curves_efficientnet_b0.png`
- `confusion_matrix_efficientnet_b0.png`
