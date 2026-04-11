"""
Optimized Baby Emotion Detector — EfficientNetB0 Transfer Learning
==================================================================
Dataset  : Happy / Sad / Neutral  (~20k images, 224×224)
Hardware : NVIDIA GeForce RTX 5060 Ti (auto-detected)
Pipeline : tf.data streaming (no RAM OOM), mixed-float16, XLA
Training : Phase-1 (frozen base) → Phase-2 (fine-tune top layers)

Run:
    cd /home/guest/bmax/imagemodel
    source .venv/bin/activate
    python optimized_emotion_model.py
"""

import os
import sys
import pathlib

# ── Bootstrap: make nvidia pip-package CUDA libs visible to TF ──────────────
# This is required when TF is in a venv and nvidia-*-cu12 packages are
# installed but the system cuda drivers are a different version.
_site_pkgs = pathlib.Path(sys.executable).parent.parent / "lib" / \
             f"python{sys.version_info.major}.{sys.version_info.minor}" / \
             "site-packages"
_nvidia_libs = ":".join(
    str(p) for p in sorted((_site_pkgs / "nvidia").glob("*/lib"))
    if p.is_dir()
)
if _nvidia_libs:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (_nvidia_libs + ":" + existing).rstrip(":")
# ── End bootstrap ────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR   = pathlib.Path("/home/guest/bmax/imagemodel/dataset")
IMG_SIZE      = (224, 224)          # EfficientNetB0 optimal input
BATCH_SIZE    = 32                  # Fits RTX 5060 Ti VRAM comfortably
PHASE1_EPOCHS = 20                  # Head-only training
PHASE2_EPOCHS = 30                  # Fine-tuning (early stop will cut this)
FINE_TUNE_AT  = 100                 # Unfreeze layers from index 100 onward
SEED          = 42
CLASSES       = ["Happy", "Neutral", "Sad"]   # alphabetical == folder order
NUM_CLASSES   = len(CLASSES)
SAVE_PATH     = "efficientnet_emotion_best.keras"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  HARDWARE SETUP  (GPU memory growth + mixed precision + XLA)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Baby Emotion Detector — Optimized Training")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓  GPU(s) detected: {[g.name for g in gpus]}")
    # Enable mixed precision (fp16 compute, fp32 weights) — ~2× faster on RTX
    mixed_precision.set_global_policy("mixed_float16")
    print("✓  Mixed precision: mixed_float16")
    # Enable XLA JIT compilation for extra kernel fusion speedup
    tf.config.optimizer.set_jit(True)
    print("✓  XLA JIT enabled")
else:
    print("⚠  No GPU detected — training will be slow on CPU")

print()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  COLLECT IMAGE PATHS & LABELS
# ─────────────────────────────────────────────────────────────────────────────
all_paths, all_labels = [], []

for label_idx, class_name in enumerate(CLASSES):
    class_dir = DATASET_DIR / class_name
    jpgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
           list(class_dir.glob("*.png"))
    all_paths.extend([str(p) for p in jpgs])
    all_labels.extend([label_idx] * len(jpgs))
    print(f"  {class_name:10s} ({label_idx}):  {len(jpgs):,} images")

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels, dtype=np.int32)
print(f"\n  Total: {len(all_paths):,} images  |  Classes: {CLASSES}\n")

# Train / val / test split  (70 / 15 / 15)
X_temp, X_test, y_temp, y_test = train_test_split(
    all_paths, all_labels, test_size=0.15, random_state=SEED, stratify=all_labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp
)   # 0.176 of 0.85 ≈ 0.15 of total → final split ≈ 70/15/15

print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLASS WEIGHTS  (correct label imbalance)
# ─────────────────────────────────────────────────────────────────────────────
cw_values = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw_values))
print(f"\n  Class weights: { {CLASSES[k]: round(v, 3) for k, v in class_weights.items()} }")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  tf.data PIPELINE  (streaming — no RAM OOM)
# ─────────────────────────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

# Augmentation layers (applied only during training, via training=True flag)
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")


def load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    """Read → decode → resize → normalize to [-1, 1] (EfficientNet convention)."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    # EfficientNetB0 expects pixels in [-1, 1]
    image = (image / 127.5) - 1.0
    return image, label


def build_dataset(paths, labels, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = build_dataset(X_train, y_train, training=True)
val_ds   = build_dataset(X_val,   y_val,   training=False)
test_ds  = build_dataset(X_test,  y_test,  training=False)

print(f"\n  Batch size: {BATCH_SIZE}  |  Steps/epoch: {len(X_train)//BATCH_SIZE}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL ARCHITECTURE  (EfficientNetB0 + custom head)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> keras.Model:
    # Pre-trained backbone — frozen for Phase 1
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)             # BN layers honour frozen state

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    # Float32 output (required when using mixed_float16)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    return model, base


model, base_model = build_model(NUM_CLASSES)
model.summary(line_length=90)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)
early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1,
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.4,
    patience=3,
    min_lr=1e-7,
    verbose=1,
)
tensorboard_cb = None  # TensorBoard not installed

callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb]

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PHASE 1 — Train classification head only
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 1: Training classification head (base frozen)")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history1 = model.fit(
    train_ds,
    epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

print(f"\n  Phase 1 best val_accuracy: {max(history1.history['val_accuracy']):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PHASE 2 — Fine-tune top layers of EfficientNetB0
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  PHASE 2: Fine-tuning layers {FINE_TUNE_AT}+ of backbone")
print("=" * 60)

base_model.trainable = True
# Freeze everything before FINE_TUNE_AT — preserve low-level features
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

print(f"  Trainable layers: {sum(l.trainable for l in base_model.layers)} / {len(base_model.layers)}")

# Recompile with a much smaller LR to avoid destroying pre-trained weights
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Reset early stopping patience for fine-tune phase
ft_callbacks = [
    ModelCheckpoint(SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=4, min_lr=1e-8, verbose=1),
]

history2 = model.fit(
    train_ds,
    epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=ft_callbacks,
    verbose=1,
)

print(f"\n  Phase 2 best val_accuracy: {max(history2.history['val_accuracy']):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION on held-out test set")
print("=" * 60)

# Load the best checkpoint
best_model = keras.models.load_model(SAVE_PATH)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
print(f"\n  Test Loss    : {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc * 100:.2f}%\n")

# Predictions for classification report + confusion matrix
y_pred_proba = best_model.predict(test_ds, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)
y_true       = np.concatenate([y for _, y in test_ds], axis=0)

print(classification_report(y_true, y_pred, target_names=CLASSES))

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

# --- Combine training histories ---
def merge(key):
    return history1.history[key] + history2.history[key]


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(merge("accuracy")) + 1)
phase1_end   = len(history1.history["accuracy"])

for ax, metric, title in zip(
    axes,
    [("accuracy", "val_accuracy"), ("loss", "val_loss")],
    ["Model Accuracy", "Model Loss"],
):
    ax.plot(epochs_range, merge(metric[0]), label=f"Train {metric[0].capitalize()}", linewidth=2)
    ax.plot(epochs_range, merge(metric[1]), label=f"Val {metric[0].capitalize()}", linewidth=2)
    ax.axvline(phase1_end, color="gray", linestyle="--", alpha=0.6, label="Phase 2 start")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("  Saved: training_curves.png")


# --- Confusion Matrix ---
def plot_confusion_matrix(cm_arr, classes, normalize=True, title="Confusion Matrix"):
    if normalize:
        cm_arr = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=12, rotation=90, va="center")
    fmt = ".2f" if normalize else "d"
    thresh = cm_arr.max() / 2.0
    for i, j in itertools.product(range(cm_arr.shape[0]), range(cm_arr.shape[1])):
        ax.text(j, i, format(cm_arr[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    plt.tight_layout()
    return fig


cm_arr = confusion_matrix(y_true, y_pred)
fig_cm = plot_confusion_matrix(cm_arr, CLASSES, normalize=True,
                                title=f"Emotion Classifier — Test Acc {test_acc*100:.1f}%")
fig_cm.savefig("emotion_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("  Saved: emotion_confusion_matrix.png")

plt.show()
print("\n✓  Training complete. Best model saved to:", SAVE_PATH)
