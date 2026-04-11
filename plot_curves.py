"""
plot_curves.py — Training curve visualiser
Reads training_history.json written by RESNET50.py and saves a PNG.
Supports any number of training phases via the phase_starts field.
Usage: python plot_curves.py
"""

import json
import os
import matplotlib.pyplot as plt

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "training_history.json")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "training_curves_resnet50.png")

# ── Load ──────────────────────────────────────────────────────────────────────
if not os.path.exists(HISTORY_FILE):
    raise FileNotFoundError(
        f"{HISTORY_FILE} not found.\n"
        "Run RESNET50.py first to generate the training history."
    )

with open(HISTORY_FILE) as f:
    h = json.load(f)

train_loss = h["train_loss"]
val_loss   = h["val_loss"]
train_acc  = h["train_acc"]
val_acc    = h["val_acc"]
n_epochs   = len(train_loss)
all_epochs = list(range(1, n_epochs + 1))

phase_starts = h.get("phase_starts", [0])
phase_labels = h.get("phase_labels", [f"Phase {i+1}" for i in range(len(phase_starts))])
backbone     = h.get("backbone", "ResNet50")

# ── Plot ──────────────────────────────────────────────────────────────────────
PHASE_COLORS = ["#d0e8ff", "#d0ffd8", "#fff0d0"]   # blue / green / orange tints

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"{backbone} Training Curves  |  Test acc: {h['test_acc']:.2f}%  |  {h['timestamp']}",
    fontsize=11,
)

# Shade each phase region
boundaries = phase_starts + [n_epochs]
for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
    color = PHASE_COLORS[i % len(PHASE_COLORS)]
    for ax in (ax_loss, ax_acc):
        ax.axvspan(start + 0.5, end + 0.5, color=color, alpha=0.5, label=phase_labels[i])
    # vertical divider between phases
    if i > 0:
        for ax in (ax_loss, ax_acc):
            ax.axvline(start + 0.5, color="grey", linewidth=0.8, linestyle="--")

# Loss
ax_loss.plot(all_epochs, train_loss, label="Train loss", color="steelblue", linewidth=1.5)
ax_loss.plot(all_epochs, val_loss,   label="Val loss",   color="tomato",    linewidth=1.5)
ax_loss.set_title("Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Cross-entropy loss")
ax_loss.legend()
ax_loss.grid(True, alpha=0.3)

# Accuracy
ax_acc.plot(all_epochs, train_acc, label="Train acc", color="steelblue", linewidth=1.5)
ax_acc.plot(all_epochs, val_acc,   label="Val acc",   color="tomato",    linewidth=1.5)
ax_acc.set_title("Accuracy (%)")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.legend()
ax_acc.grid(True, alpha=0.3)

# Phase labels at top of each shaded band
for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
    mid = (start + end) / 2 + 0.5
    for ax in (ax_loss, ax_acc):
        ax.text(mid, ax.get_ylim()[1], phase_labels[i],
                ha="center", va="top", fontsize=7.5, color="#444444")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
print(f"Saved: {OUTPUT_FILE}")
plt.show()
