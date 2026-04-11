"""
Emotion classifier — EfficientNet-B0 backbone
Improvements over the original ResNet-50 script:
  1. EfficientNet-B0 backbone (5M params vs 23M — less overfitting on 20K images)
  2. 96×96 grayscale input  (was 224×224 fake-RGB — 4.67× upscale caused blur)
  3. First conv adapted to 1 channel (pretrained weights averaged across RGB)
  4. Class-weighted loss + label smoothing 0.1  (corrects Happy/Neutral/Sad imbalance)
  5. Stronger augmentation: rotation, affine, random erasing
  6. AdamW optimiser with weight decay
  7. 3-phase fine-tuning: head → last 30 layers → last 60 layers
  8. Training history saved to training_history.json for plot_curves.py
"""

import os, glob, json, itertools, datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from PIL import Image

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/guest/bmax/imagemodel/dataset"
CLASSES  = ["Happy", "Neutral", "Sad"]
IMG_SIZE = 96       # 2× upscale from native 48×48 — far less blur than 224×224
BATCH    = 64       # larger batch fits easily at 96×96

# Actual per-pixel mean/std of this grayscale dataset
MEAN = 0.4889
STD  = 0.2521

# ── Collect paths + labels ────────────────────────────────────────────────────
all_paths, all_labels = [], []
for idx, cls in enumerate(CLASSES):
    paths = glob.glob(os.path.join(DATA_DIR, cls, "*.jpg"))
    all_paths.extend(paths)
    all_labels.extend([idx] * len(paths))
    print(f"{cls}: {len(paths)}")

# ── Split: 81% train / 9% val / 10% test ─────────────────────────────────────
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.1, random_state=42, stratify=all_labels
)
train_paths, val_paths, ytrain, yval = train_test_split(
    train_val_paths, train_val_labels, test_size=0.1, random_state=42,
    stratify=train_val_labels
)
print(f"\nSplit → train: {len(train_paths)}  val: {len(val_paths)}  test: {len(test_paths)}")

# ── Class weights (correct for Happy being over-represented) ─────────────────
class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(ytrain))
print(f"Class weights: { {c: f'{w:.3f}' for c, w in zip(CLASSES, class_weights)} }")

# ── Dataset ───────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load as grayscale (1 channel) — no fake-RGB conversion
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img), self.labels[idx]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([MEAN], [STD]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),   # simulate occlusion
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([MEAN], [STD]),
])

train_loader = DataLoader(EmotionDataset(train_paths, ytrain, train_tf),
                          batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(EmotionDataset(val_paths,   yval,   eval_tf),
                          batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(EmotionDataset(test_paths,  test_labels, eval_tf),
                          batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

# ── Model — EfficientNet-B0 with single-channel input ─────────────────────────
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Adapt first conv from 3-channel RGB → 1-channel grayscale.
# Average pretrained RGB weights so we don't throw away ImageNet knowledge.
first_conv = model.features[0][0]
model.features[0][0] = nn.Conv2d(
    1, first_conv.out_channels,
    kernel_size=first_conv.kernel_size,
    stride=first_conv.stride,
    padding=first_conv.padding,
    bias=False,
)
with torch.no_grad():
    model.features[0][0].weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))

# Freeze entire backbone
for p in model.parameters():
    p.requires_grad = False

# Replace classifier head
model.classifier = nn.Sequential(
    nn.BatchNorm1d(1280),
    nn.Dropout(0.4),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(CLASSES)),
)

model = model.to(DEVICE)

# ── Loss / scaler ─────────────────────────────────────────────────────────────
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
scaler    = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

# ── Training helpers ──────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_correct = total = 0
    ctx = torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda")
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            with ctx:
                logits = model(imgs)
                loss   = criterion(logits, labels)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss    += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += len(labels)
    return total_loss / total, total_correct / total * 100


def train_phase(model, optimizer, epochs, desc, patience=5):
    print(f"\n{desc}")
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    best_val_loss = float("inf")
    pat_count     = 0
    best_state    = None
    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"  Epoch {epoch:>2}/{epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.1f}%  "
              f"val loss={va_loss:.4f} acc={va_acc:.1f}%")
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            pat_count     = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history


def unfreeze_last_n(model, n):
    """Unfreeze the last n parameter-containing modules (leaves only)."""
    for p in model.parameters():
        p.requires_grad = False
    param_mods = [m for m in model.modules() if list(m.parameters(recurse=False))]
    for m in param_mods[-n:]:
        for p in m.parameters(recurse=False):
            p.requires_grad = True


# ── Phase 1: head only ────────────────────────────────────────────────────────
unfreeze_last_n(model, 0)   # all frozen; classifier params are new so always trained
for p in model.classifier.parameters():
    p.requires_grad = True

opt1   = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
hist1  = train_phase(model, opt1, epochs=15,
                     desc="Phase 1: head only (backbone frozen)...")

# ── Phase 2: unfreeze last 30 layers ─────────────────────────────────────────
unfreeze_last_n(model, 30)
opt2  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-4, weight_decay=1e-4)
hist2 = train_phase(model, opt2, epochs=20,
                    desc="Phase 2: fine-tuning last 30 layers...")

# ── Phase 3: unfreeze last 60 layers ─────────────────────────────────────────
unfreeze_last_n(model, 60)
opt3  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-5, weight_decay=1e-4)
hist3 = train_phase(model, opt3, epochs=20,
                    desc="Phase 3: fine-tuning last 60 layers...")

# ── Evaluate ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix"):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar()
    ticks = range(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    fmt    = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True label"); plt.xlabel("Predicted label")

model.eval()
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
print(classification_report(all_true, all_preds, target_names=CLASSES))

# ── Save model ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "resnet50_emotion.pth")
print("Saved: resnet50_emotion.pth")

# ── Merge phase histories ─────────────────────────────────────────────────────
def merge(*hists):
    out = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    boundaries = []
    for h in hists:
        boundaries.append(len(out["train_loss"]))
        for k in out:
            out[k].extend(h[k])
    return out, boundaries

merged, phase_starts = merge(hist1, hist2, hist3)
timestamp = datetime.datetime.now().isoformat(timespec="seconds")

history_data = {
    "timestamp":    timestamp,
    "backbone":     "EfficientNet-B0",
    "img_size":     IMG_SIZE,
    "classes":      CLASSES,
    "phase_starts": phase_starts,
    "phase_labels": ["Head only", "Fine-tune 30", "Fine-tune 60"],
    **merged,
    "test_acc":     float(acc),
}
with open("training_history.json", "w") as f:
    json.dump(history_data, f, indent=2)
print("Saved: training_history.json")

# ── Plot: confusion matrix ────────────────────────────────────────────────────
def _plot_cm_ax(ax, cm, classes, normalize, title):
    data = cm.astype("float") / cm.sum(axis=1, keepdims=True) if normalize else cm.copy()
    ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=11)
    ticks = range(len(classes))
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(classes)
    fmt    = ".2f" if normalize else "d"
    thresh = data.max() / 2.0
    for i, j in itertools.product(range(data.shape[0]), range(data.shape[1])):
        ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                color="white" if data[i, j] > thresh else "black", fontsize=10)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")

cm = confusion_matrix(all_true, all_preds)
fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig_cm.suptitle(f"EfficientNet-B0 Confusion Matrix  |  Test acc: {acc:.2f}%", fontsize=12)
_plot_cm_ax(ax1, cm, CLASSES, normalize=False, title="Raw counts")
_plot_cm_ax(ax2, cm, CLASSES, normalize=True,  title="Normalised (recall per class)")
plt.tight_layout()
plt.savefig("confusion_matrix_resnet50.png", dpi=150)
print("Saved: confusion_matrix_resnet50.png")
plt.show()

# ── Plot: training curves ─────────────────────────────────────────────────────
train_loss   = merged["train_loss"]
val_loss     = merged["val_loss"]
train_acc    = merged["train_acc"]
val_acc      = merged["val_acc"]
n_epochs     = len(train_loss)
all_epochs   = list(range(1, n_epochs + 1))
phase_labels = ["Head only", "Fine-tune 30", "Fine-tune 60"]
boundaries   = phase_starts + [n_epochs]
phase_colors = ["#d0e8ff", "#d0ffd8", "#fff0d0"]

fig_c, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
fig_c.suptitle(
    f"EfficientNet-B0 Training Curves  |  Test acc: {acc:.2f}%  |  {timestamp}",
    fontsize=11,
)

for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
    for ax in (ax_loss, ax_acc):
        ax.axvspan(start + 0.5, end + 0.5,
                   color=phase_colors[i % len(phase_colors)], alpha=0.5,
                   label=phase_labels[i])
    if i > 0:
        for ax in (ax_loss, ax_acc):
            ax.axvline(start + 0.5, color="grey", linewidth=0.8, linestyle="--")

ax_loss.plot(all_epochs, train_loss, label="Train loss", color="steelblue", linewidth=1.5)
ax_loss.plot(all_epochs, val_loss,   label="Val loss",   color="tomato",    linewidth=1.5)
ax_loss.set_title("Loss"); ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
ax_loss.legend(); ax_loss.grid(True, alpha=0.3)

ax_acc.plot(all_epochs, train_acc, label="Train acc", color="steelblue", linewidth=1.5)
ax_acc.plot(all_epochs, val_acc,   label="Val acc",   color="tomato",    linewidth=1.5)
ax_acc.set_title("Accuracy"); ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy (%)")
ax_acc.legend(); ax_acc.grid(True, alpha=0.3)

for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
    mid = (start + end) / 2 + 0.5
    for ax in (ax_loss, ax_acc):
        ax.text(mid, ax.get_ylim()[1], phase_labels[i],
                ha="center", va="top", fontsize=7.5, color="#444444")

plt.tight_layout()
plt.savefig("training_curves_resnet50.png", dpi=150)
print("Saved: training_curves_resnet50.png")
plt.show()
