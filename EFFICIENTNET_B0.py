"""
Emotion classifier — EfficientNet-B0 backbone
v4 optimisations over v3:
  1. IMG_SIZE=112 — better detail from higher-res AffectNet/RAFDB images
  2. ImageNet normalisation stats — correct for pretrained backbone
  3. Richer augmentation: GaussianBlur + RandomGrayscale
  4. MixUp (α=0.2) — strongest single regulariser for image classification
  5. FocalLoss with label smoothing (ε=0.1) — prevents overconfidence
  6. Cosine LR with linear warmup per phase — smoother than ReduceLROnPlateau
  7. Gradient clipping (max_norm=1.0) — stability on mixed-quality dataset
  8. EMA weights (decay=0.9995) — used for val/test, typically +0.5–1% over raw model
  9. Simplified head: Dropout(0.5) → Linear(1280, 3) — 3 classes need no bottleneck
"""

import copy, os, glob, json, itertools, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
DATA_DIR    = "/home/guest/bmax/imagemodel/dataset"
CLASSES     = ["Happy", "Neutral", "Sad"]
IMG_SIZE    = 112
BATCH       = 64
MIXUP_ALPHA = 0.2

# ImageNet stats — correct for pretrained EfficientNet-B0 backbone
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

RESUME_FROM = None  # training from scratch with updated architecture

# ── Collect paths + labels ────────────────────────────────────────────────────
all_paths, all_labels = [], []
for idx, cls in enumerate(CLASSES):
    paths = (
        glob.glob(os.path.join(DATA_DIR, cls, "*.jpg"))  +
        glob.glob(os.path.join(DATA_DIR, cls, "*.jpeg")) +
        glob.glob(os.path.join(DATA_DIR, cls, "*.png"))
    )
    all_paths.extend(paths)
    all_labels.extend([idx] * len(paths))
    print(f"{cls}: {len(paths)}")

# ── Split ─────────────────────────────────────────────────────────────────────
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
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),     # robustness to FER-2013 greyscale origin
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

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

# ── Focal Loss with label smoothing ──────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, weight=self.alpha,
                               label_smoothing=self.label_smoothing, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()

# ── EMA ───────────────────────────────────────────────────────────────────────
class EMA:
    """Exponential moving average of model weights for stable evaluation."""
    def __init__(self, model, decay=0.9995):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.data, 1 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

for p in model.parameters():
    p.requires_grad = False

# Simplified head — 3 classes don't benefit from a 256-dim bottleneck
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(1280, len(CLASSES)),
)
model = model.to(DEVICE)

ema = EMA(model, decay=0.9995)

# ── Loss / scaler ─────────────────────────────────────────────────────────────
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion      = FocalLoss(alpha=weights_tensor, gamma=2.0, label_smoothing=0.1)
scaler         = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

# ── Training helpers ──────────────────────────────────────────────────────────
def run_train_epoch(model, loader, optimizer):
    model.train()
    total_loss = total_correct = total = 0
    ctx = torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda")

    with torch.enable_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # MixUp
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            idx = torch.randperm(imgs.size(0), device=DEVICE)
            imgs_mixed = lam * imgs + (1 - lam) * imgs[idx]
            labels_b   = labels[idx]

            optimizer.zero_grad()
            with ctx:
                logits = model(imgs_mixed)
                loss   = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss    += loss.item() * len(labels)
            # accuracy measured on un-mixed logits (informational only)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += len(labels)

    return total_loss / total, total_correct / total * 100


def run_eval_epoch(model, loader):
    """Evaluate using the EMA model."""
    model.eval()
    total_loss = total_correct = total = 0
    ctx = torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda")

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with ctx:
                logits = model(imgs)
                loss   = criterion(logits, labels)
            total_loss    += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += len(labels)

    return total_loss / total, total_correct / total * 100


def make_scheduler(optimizer, lr, epochs, warmup_epochs=2):
    """Linear warmup for warmup_epochs then cosine decay to lr/100."""
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1),
                               eta_min=lr / 100)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def unfreeze_last_n(model, n):
    for p in model.parameters():
        p.requires_grad = False
    param_mods = [m for m in model.modules() if list(m.parameters(recurse=False))]
    for m in param_mods[-n:]:
        for p in m.parameters(recurse=False):
            p.requires_grad = True


def train_phase(model, lr, epochs, desc, patience=8):
    print(f"\n{desc}")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler     = make_scheduler(optimizer, lr, epochs)
    best_val_loss = float("inf")
    pat_count     = 0
    best_state    = None
    best_ema      = None
    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_train_epoch(model, train_loader, optimizer)
        va_loss, va_acc = run_eval_epoch(ema.model, val_loader)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"  Epoch {epoch:>2}/{epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.1f}%  "
              f"val loss={va_loss:.4f} acc={va_acc:.1f}%  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            pat_count     = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ema      = {k: v.cpu().clone() for k, v in ema.model.state_dict().items()}
        else:
            pat_count += 1
            if pat_count >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    if best_ema:
        ema.model.load_state_dict(best_ema)
    return history

# ── Training ──────────────────────────────────────────────────────────────────
# Phase 1: head only
for p in model.classifier.parameters():
    p.requires_grad = True
hist1 = train_phase(model, lr=1e-3, epochs=15, patience=5,
                    desc="Phase 1: head only (backbone frozen)...")

# Phase 2: unfreeze last 30 layers
unfreeze_last_n(model, 30)
hist2 = train_phase(model, lr=1e-4, epochs=25, patience=8,
                    desc="Phase 2: fine-tuning last 30 layers...")

# Phase 3: unfreeze last 60 layers
unfreeze_last_n(model, 60)
hist3 = train_phase(model, lr=1e-5, epochs=50, patience=12,
                    desc="Phase 3: fine-tuning last 60 layers...")

all_hists    = [hist1, hist2, hist3]
phase_starts = [0, len(hist1["train_loss"]),
                len(hist1["train_loss"]) + len(hist2["train_loss"])]
phase_labels = ["Head only", "Fine-tune 30", "Fine-tune 60"]

# ── Evaluate (EMA model) ──────────────────────────────────────────────────────
ema.model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        preds = ema.model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
acc = (all_preds == all_true).mean() * 100

print(f"\nTest Accuracy (EMA): {acc:.2f}%")
print(classification_report(all_true, all_preds, target_names=CLASSES))

# ── Save model (EMA weights) ──────────────────────────────────────────────────
torch.save(ema.model.state_dict(), "efficientnet_b0_emotion.pth")
print("Saved: efficientnet_b0_emotion.pth")

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
plt.savefig("confusion_matrix_efficientnet_b0.png", dpi=150)
print("Saved: confusion_matrix_efficientnet_b0.png")
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
plt.savefig("training_curves_efficientnet_b0.png", dpi=150)
print("Saved: training_curves_efficientnet_b0.png")
plt.show()
