# EfficientNet-B0 Emotion Recognition — FER-2013, Balanced AffectNet & RAFDB

Testing and fine-tuning of different EfficientNet-B0 based models for emotion recognition on the FER-2013, Balanced AffectNet, and RAFDB datasets.

Each version folder contains the training script, model weights, plots, and metrics from that training run.

## Versions

| Folder | Backbone | Dataset | Test Accuracy | Date |
|---|---|---|---|---|
| `v1_2026-04-11_19-42/` | EfficientNet-B0 | FER-2013 | 71.85% | 2026-04-11 |
| `v2_2026-04-11_20-42/` | EfficientNet-B0 + Focal Loss + Balanced Sampler | FER-2013 | 73.80% | 2026-04-11 |
| `v3_2026-04-18_16-52/` | EfficientNet-B0 + Focal Loss + Balanced Sampler + RGB | FER-2013 + Balanced AffectNet + RAFDB | 76.00% | 2026-04-18 |
| `v4_2026-04-18_17-21/` | EfficientNet-B0 + MixUp + Label Smoothing + EMA + Cosine LR | FER-2013 + Balanced AffectNet + RAFDB | 77.65% | 2026-04-18 |

## Datasets

**FER-2013** — 48×48 grayscale facial expression images, 3 classes: **Happy**, **Neutral**, **Sad**.

**Balanced AffectNet** — Higher resolution facial expression images with balanced class distribution.

**RAFDB** (Real-world Affective Faces Database) — Real-world facial expression images collected from the internet, higher diversity than lab-collected datasets. Introduced from v3 onwards.
