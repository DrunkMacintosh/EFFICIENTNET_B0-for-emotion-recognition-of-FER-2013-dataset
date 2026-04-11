# ResNet-50 Emotion Recognition — FER-2013

Testing and fine-tuning of different ResNet-50 based models for emotion recognition on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset.

Each version folder contains the training script, model weights, plots, and metrics from that training run.

## Versions

| Folder | Backbone | Test Accuracy | Date |
|---|---|---|---|
| `v1_2026-04-11_19-42/` | EfficientNet-B0 | 71.85% | 2026-04-11 |
| `v2_2026-04-11_20-42/` | EfficientNet-B0 + Focal Loss + Balanced Sampler | 73.80% | 2026-04-11 |

## Dataset

FER-2013 — 48×48 grayscale facial expression images, 3 classes: **Happy**, **Neutral**, **Sad**.
