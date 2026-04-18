"""
Emotion classifier — InceptionResNetV1 backbone (VGGFace2 pretrained)
Face-pretrained backbone: 3.3M faces, 9,131 identities (VGGFace2)

v6 fine-tuning changes over v5:
  1. Resume from v5 EMA weights — continue from 86.68%, no wasted epochs
  2. CutMix (50/50 with MixUp) — cuts a face region and pastes from another image,
     preserving local spatial structure better than global pixel blending
  3. TTA (test-time augmentation) — averages original + horizontal flip at inference
  4. Label smoothing 0.1 → 0.05 — model is well-trained; less smoothing = more precision
  5. Dropout 0.5 → 0.3 — EMA handles regularisation; polishing, not learning from scratch
  6. Single continuation phase at LR=5e-6 — gentle adjustments on a converged model
"""

import copy, os, glob, json, itertools, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from facenet_pytorch import InceptionResnetV1
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
IMG_SIZE    = 160          # native input size for InceptionResNetV1
BATCH        = 64
MIXUP_ALPHA  = 0.2
CUTMIX_ALPHA = 1.0   # CutMix lambda drawn from Beta(1,1) = Uniform(0,1)

# Resume from v5 EMA weights — skip 3-phase from scratch
RESUME_FROM = "v5_2026-04-18_22-30/face_emotion.pth"

# Normalization for InceptionResNetV1: maps [0,1] → [-1,1]
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

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
    transforms.RandomGrayscale(p=0.1),
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
class FaceEmotionNet(nn.Module):
    """InceptionResNetV1 face backbone + lightweight emotion head."""
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        # Pretrained on VGGFace2: 3.3M images, 9,131 identities
        # classify=False → returns raw 512-dim face embeddings
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

model = FaceEmotionNet(num_classes=len(CLASSES)).to(DEVICE)

# Freeze entire backbone; only head trains in phase 1
for p in model.backbone.parameters():
    p.requires_grad = False

ema = EMA(model, decay=0.9995)

# ── Loss / scaler ─────────────────────────────────────────────────────────────
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion      = FocalLoss(alpha=weights_tensor, gamma=2.0, label_smoothing=0.05)
scaler         = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

# ── CutMix ────────────────────────────────────────────────────────────────────
def apply_cutmix(imgs, labels):
    """Paste a random rectangular region from one image into another."""
    lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
    B, C, H, W = imgs.shape
    idx = torch.randperm(B, device=imgs.device)

    cut_h = int(H * np.sqrt(1.0 - lam))
    cut_w = int(W * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)

    imgs = imgs.clone()
    imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)   # actual mix ratio after clipping
    return imgs, labels, labels[idx], lam


# ── Training helpers ──────────────────────────────────────────────────────────
def run_train_epoch(model, loader, optimizer):
    model.train()
    total_loss = total_correct = total = 0
    ctx = torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda")

    with torch.enable_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # Randomly apply CutMix or MixUp with equal probability
            if np.random.rand() < 0.5:
                imgs, labels_a, labels_b, lam = apply_cutmix(imgs, labels)
            else:
                lam      = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                idx      = torch.randperm(imgs.size(0), device=DEVICE)
                imgs     = lam * imgs + (1 - lam) * imgs[idx]
                labels_a, labels_b = labels, labels[idx]

            optimizer.zero_grad()
            with ctx:
                logits = model(imgs)
                loss   = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss    += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels_a).sum().item()
            total         += len(labels)

    return total_loss / total, total_correct / total * 100


def run_eval_epoch(model, loader, tta=False):
    """Evaluate with optional TTA: average logits over original + horizontal flip."""
    model.eval()
    total_loss = total_correct = total = 0
    ctx = torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda")

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with ctx:
                logits = model(imgs)
                if tta:
                    logits = (logits + model(torch.flip(imgs, dims=[3]))) * 0.5
                loss = criterion(logits, labels)
            total_loss    += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += len(labels)

    return total_loss / total, total_correct / total * 100


def make_scheduler(optimizer, lr, epochs, warmup_epochs=2):
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
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler     = make_scheduler(optimizer, lr, epochs)
    best_val_loss = float("inf")
    pat_count     = 0
    best_state    = None
    best_ema      = None
    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_train_epoch(model, train_loader, optimizer)
        va_loss, va_acc = run_eval_epoch(ema.model, val_loader, tta=True)
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
if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"\nResuming from {RESUME_FROM}")
    state = torch.load(RESUME_FROM, map_location=DEVICE)
    model.load_state_dict(state)
    # Re-initialise EMA from the loaded weights so it tracks from v5 quality
    ema = EMA(model, decay=0.9995)
    print("Loaded v5 EMA weights into model and EMA.")

# All layers unfrozen — gentle full fine-tuning at very low LR
for p in model.parameters():
    p.requires_grad = True

hist1 = train_phase(model, lr=5e-6, epochs=40, patience=15,
                    desc="Continued fine-tuning (all layers, CutMix+MixUp, TTA val)...")

all_hists    = [hist1]
phase_starts = [0]
phase_labels = ["Continued fine-tuning (from v5)"]

# ── Evaluate with TTA (EMA model) ────────────────────────────────────────────
ema.model.eval()
all_preds, all_true = [], []
ctx = torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda")
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        with ctx:
            logits = (ema.model(imgs) + ema.model(torch.flip(imgs, dims=[3]))) * 0.5
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_true.extend(labels.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
acc = (all_preds == all_true).mean() * 100

print(f"\nTest Accuracy (EMA): {acc:.2f}%")
print(classification_report(all_true, all_preds, target_names=CLASSES))

# ── Save model (EMA weights) ──────────────────────────────────────────────────
torch.save(ema.model.state_dict(), "face_emotion.pth")
print("Saved: face_emotion.pth")

# ── Merge histories + save JSON ───────────────────────────────────────────────
merged = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
for h in all_hists:
    for k in merged:
        merged[k].extend(h[k])

timestamp = datetime.datetime.now().isoformat(timespec="seconds")
history_data = {
    "timestamp":    timestamp,
    "backbone":     "InceptionResNetV1-VGGFace2",
    "img_size":     IMG_SIZE,
    "classes":      CLASSES,
    "phase_starts": phase_starts,
    "phase_labels": phase_labels,
    **merged,
    "test_acc":     float(acc),
}
with open("face_training_history.json", "w") as f:
    json.dump(history_data, f, indent=2)
print("Saved: face_training_history.json")

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
fig_cm.suptitle(f"InceptionResNetV1 (VGGFace2) Confusion Matrix  |  Test acc: {acc:.2f}%",
                fontsize=12)
_plot_cm_ax(ax1, cm, CLASSES, normalize=False, title="Raw counts")
_plot_cm_ax(ax2, cm, CLASSES, normalize=True,  title="Normalised (recall per class)")
plt.tight_layout()
plt.savefig("confusion_matrix_face_emotion.png", dpi=150)
print("Saved: confusion_matrix_face_emotion.png")
plt.show()

# ── Plot: training curves ─────────────────────────────────────────────────────
n_epochs     = len(merged["train_loss"])
all_epochs   = list(range(1, n_epochs + 1))
boundaries   = phase_starts + [n_epochs]
phase_colors = ["#d0e8ff", "#d0ffd8", "#fff0d0"]

fig_c, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
fig_c.suptitle(
    f"InceptionResNetV1 (VGGFace2) Training Curves  |  Test acc: {acc:.2f}%  |  {timestamp}",
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
plt.savefig("training_curves_face_emotion.png", dpi=150)
print("Saved: training_curves_face_emotion.png")
plt.show()
