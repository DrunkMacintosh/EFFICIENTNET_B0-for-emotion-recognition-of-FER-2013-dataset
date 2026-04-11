"""
Emotion classifier — EfficientNet-B0 backbone
v2 improvements over v1:
  1. Focal Loss (γ=2) — down-weights easy Happy examples, forces focus on hard Neutral/Sad
  2. WeightedRandomSampler — every batch is class-balanced regardless of loss function
  3. Checkpoint resume — loads v1 weights and continues fine-tuning (no wasted epochs)
  4. Longer training + higher patience — model was still converging at end of v1
"""

import os, glob, json, itertools, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
DATA_DIR     = "/home/guest/bmax/imagemodel/dataset"
CLASSES      = ["Happy", "Neutral", "Sad"]
IMG_SIZE     = 96
BATCH        = 64
MEAN, STD    = 0.4889, 0.2521

# Set to a .pth path to continue from a checkpoint instead of training from scratch.
# Skips phases 1 & 2 and goes straight to full fine-tuning with the new loss.
RESUME_FROM  = "v1_2026-04-11_19-42/resnet50_emotion.pth"

# ── Collect paths + labels ────────────────────────────────────────────────────
all_paths, all_labels = [], []
for idx, cls in enumerate(CLASSES):
    paths = glob.glob(os.path.join(DATA_DIR, cls, "*.jpg"))
    all_paths.extend(paths)
    all_labels.extend([idx] * len(paths))
    print(f"{cls}: {len(paths)}")

# ── Split (same seed as v1 — identical test set for fair comparison) ──────────
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.1, random_state=42, stratify=all_labels
)
train_paths, val_paths, ytrain, yval = train_test_split(
    train_val_paths, train_val_labels, test_size=0.1, random_state=42,
    stratify=train_val_labels
)
print(f"\nSplit → train: {len(train_paths)}  val: {len(val_paths)}  test: {len(test_paths)}")

# ── Class weights ─────────────────────────────────────────────────────────────
class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]),
                                     y=np.array(ytrain))
print(f"Class weights: { {c: f'{w:.3f}' for c, w in zip(CLASSES, class_weights)} }")

# ── Dataset ───────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img), self.labels[idx]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([MEAN], [STD]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([MEAN], [STD]),
])

# WeightedRandomSampler — each batch drawn with equal class probability,
# so Neutral and Sad are no longer under-represented in every batch.
sample_weights = torch.tensor([class_weights[l] for l in ytrain], dtype=torch.float32)
sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                       replacement=True)

train_loader = DataLoader(EmotionDataset(train_paths, ytrain, train_tf),
                          batch_size=BATCH, sampler=sampler,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(EmotionDataset(val_paths,   yval,   eval_tf),
                          batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(EmotionDataset(test_paths,  test_labels, eval_tf),
                          batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    gamma=2 is the standard value from the original paper.
    alpha=class_weights corrects for class imbalance on top of the focal term.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha   # class weight tensor
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)                        # probability of correct class
        loss = (1 - pt) ** self.gamma * ce           # down-weight easy examples
        return loss.mean()

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

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

for p in model.parameters():
    p.requires_grad = False

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
criterion      = FocalLoss(alpha=weights_tensor, gamma=2.0)
scaler         = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

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


def unfreeze_last_n(model, n):
    for p in model.parameters():
        p.requires_grad = False
    param_mods = [m for m in model.modules() if list(m.parameters(recurse=False))]
    for m in param_mods[-n:]:
        for p in m.parameters(recurse=False):
            p.requires_grad = True


def train_phase(model, optimizer, epochs, desc, patience=8):
    print(f"\n{desc}")
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
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

# ── Training ──────────────────────────────────────────────────────────────────
if RESUME_FROM and os.path.exists(RESUME_FROM):
    # ── Resume from checkpoint ────────────────────────────────────────────────
    # Load v1 weights — skip phases 1 & 2, go straight to full fine-tuning
    # with Focal Loss + WeightedRandomSampler to target Neutral and Sad.
    print(f"\nResuming from {RESUME_FROM}")
    model.load_state_dict(torch.load(RESUME_FROM, map_location=DEVICE))

    unfreeze_last_n(model, 60)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=5e-6, weight_decay=1e-4)
    hist_resume = train_phase(model, opt, epochs=50,
                              desc="Continued fine-tuning — Focal Loss + balanced sampler "
                                   "(all 60 layers, lr=5e-6)...",
                              patience=10)

    all_hists    = [hist_resume]
    phase_starts = [0]
    phase_labels = ["Focal fine-tune (from v1)"]

else:
    # ── Full training from scratch ────────────────────────────────────────────
    for p in model.classifier.parameters():
        p.requires_grad = True
    opt1  = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    hist1 = train_phase(model, opt1, epochs=15, patience=5,
                        desc="Phase 1: head only (backbone frozen)...")

    unfreeze_last_n(model, 30)
    opt2  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=1e-4, weight_decay=1e-4)
    hist2 = train_phase(model, opt2, epochs=30, patience=8,
                        desc="Phase 2: fine-tuning last 30 layers...")

    unfreeze_last_n(model, 60)
    opt3  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=1e-5, weight_decay=1e-4)
    hist3 = train_phase(model, opt3, epochs=40, patience=10,
                        desc="Phase 3: fine-tuning last 60 layers...")

    all_hists    = [hist1, hist2, hist3]
    phase_starts = [0, len(hist1["train_loss"]),
                    len(hist1["train_loss"]) + len(hist2["train_loss"])]
    phase_labels = ["Head only", "Fine-tune 30", "Fine-tune 60"]

# ── Evaluate ──────────────────────────────────────────────────────────────────
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

# ── Merge histories + save JSON ───────────────────────────────────────────────
merged = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
for h in all_hists:
    for k in merged:
        merged[k].extend(h[k])

timestamp = datetime.datetime.now().isoformat(timespec="seconds")
history_data = {
    "timestamp":    timestamp,
    "backbone":     "EfficientNet-B0",
    "img_size":     IMG_SIZE,
    "classes":      CLASSES,
    "phase_starts": phase_starts,
    "phase_labels": phase_labels,
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
n_epochs     = len(merged["train_loss"])
all_epochs   = list(range(1, n_epochs + 1))
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

ax_loss.plot(all_epochs, merged["train_loss"], label="Train loss", color="steelblue", linewidth=1.5)
ax_loss.plot(all_epochs, merged["val_loss"],   label="Val loss",   color="tomato",    linewidth=1.5)
ax_loss.set_title("Loss"); ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
ax_loss.legend(); ax_loss.grid(True, alpha=0.3)

ax_acc.plot(all_epochs, merged["train_acc"], label="Train acc", color="steelblue", linewidth=1.5)
ax_acc.plot(all_epochs, merged["val_acc"],   label="Val acc",   color="tomato",    linewidth=1.5)
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
