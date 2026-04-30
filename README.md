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
| `v5_2026-04-18_22-30/` | InceptionResNetV1 (VGGFace2 face-pretrained) | FER-2013 + Balanced AffectNet + RAFDB | 86.68% | 2026-04-18 |
| `v6_2026-04-19_00-12/` | InceptionResNetV1 + CutMix + TTA (fine-tuned from v5) | FER-2013 + Balanced AffectNet + RAFDB | 87.25% | 2026-04-19 |
| `v7_2026-04-30_20-28/` | InceptionResNetV1 + CutMix + TTA (fine-tuned from v6) | FER-2013 + Balanced AffectNet + RAFDB | 87.77% | 2026-04-30 |

## Datasets

**FER-2013** — 48×48 grayscale facial expression images, 3 classes: **Happy**, **Neutral**, **Sad**.

**Balanced AffectNet** — Higher resolution facial expression images with balanced class distribution.

**RAFDB** (Real-world Affective Faces Database) — Real-world facial expression images collected from the internet, higher diversity than lab-collected datasets. Introduced from v3 onwards.
