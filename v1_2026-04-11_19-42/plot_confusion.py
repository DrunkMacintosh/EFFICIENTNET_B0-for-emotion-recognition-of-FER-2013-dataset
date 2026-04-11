"""
plot_confusion.py — Confusion matrix visualiser
Loads resnet50_emotion.pth, rebuilds the same test split, runs inference,
and saves a PNG.
Usage: python plot_confusion.py
"""

import os
import glob
import itertools
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PIL import Image

MODEL_FILE  = os.path.join(os.path.dirname(__file__), "resnet50_emotion.pth")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "confusion_matrix_resnet50.png")
DATA_DIR    = "/home/guest/bmax/imagemodel/dataset"
CLASSES     = ["Happy", "Neutral", "Sad"]
IMG_SIZE    = 96
MEAN, STD   = 0.4889, 0.2521
BATCH       = 64

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Rebuild the same test split (random_state=42 matches RESNET50.py) ─────────
all_paths, all_labels = [], []
for idx, cls in enumerate(CLASSES):
    paths = glob.glob(os.path.join(DATA_DIR, cls, "*.jpg"))
    all_paths.extend(paths)
    all_labels.extend([idx] * len(paths))

train_val_paths, test_paths, _, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.1, random_state=42
)
print(f"Test set: {len(test_paths)} images")

# ── Dataset / loader ──────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")   # grayscale
        return self.transform(img), self.labels[idx]

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([MEAN], [STD]),
])

test_loader = DataLoader(EmotionDataset(test_paths, test_labels, eval_tf),
                         batch_size=BATCH, shuffle=False,
                         num_workers=4, pin_memory=True)

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} not found.\nRun RESNET50.py first to train and save the model."
    )

model = models.efficientnet_b0(weights=None)
first_conv = model.features[0][0]
model.features[0][0] = nn.Conv2d(
    1, first_conv.out_channels,
    kernel_size=first_conv.kernel_size,
    stride=first_conv.stride,
    padding=first_conv.padding,
    bias=False,
)
model.classifier = nn.Sequential(
    nn.BatchNorm1d(1280),
    nn.Dropout(0.4),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(CLASSES)),
)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

# ── Inference ─────────────────────────────────────────────────────────────────
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

acc = (all_preds == all_true).mean() * 100
print(f"\nTest Accuracy: {acc:.2f}%")
print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=CLASSES))

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_cm(ax, cm, classes, normalize, title):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=11)
    ticks = range(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(classes)
    fmt    = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=10)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

cm = confusion_matrix(all_true, all_preds)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"ResNet50 Confusion Matrix  |  Test acc: {acc:.2f}%", fontsize=12)

plot_cm(ax1, cm.copy(), CLASSES, normalize=False, title="Raw counts")
plot_cm(ax2, cm.copy(), CLASSES, normalize=True,  title="Normalised (recall per class)")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
print(f"\nSaved: {OUTPUT_FILE}")
plt.show()
