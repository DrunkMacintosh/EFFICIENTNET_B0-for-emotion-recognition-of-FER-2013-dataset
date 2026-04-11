"""
Improved Baby Emotion Detector  v2 — EfficientNetV2S
=====================================================
Key improvements over v1 (EfficientNetB0):

  1. EfficientNetV2S backbone  — 21.5 M params vs 5.3 M; ImageNet top-1 ~84 %
  2. Fixed preprocessing bug   — v1 passed images as [-1, 1] to EfficientNetB0,
                                  which already rescales internally from [0, 255].
                                  v2 passes [0, 255] and sets include_preprocessing=True.
  3. Label smoothing ε = 0.1   — prevents overconfidence; improves calibration
  4. Cosine-decay LR per phase — smoother convergence; avoids sharp LR steps
  5. Stronger augmentation     — wider random ranges for all transforms
  6. 3-phase fine-tuning       — head-only → top-100 backbone layers → top-200
  7. Larger head               — 512 → 256 neurons with L2 regularisation

Run:
    cd /home/guest/bmax/imagemodel
    . .venv/bin/activate
    NVIDIA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -name lib -type d | sort | tr '\\n' ':')
    LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}" python3 improved_emotion_model.py
"""

import os
import sys
import pathlib
import itertools

# ── Bootstrap: make nvidia pip-package CUDA libs visible to TF ──────────────
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR    = pathlib.Path("/home/guest/bmax/imagemodel/dataset")
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
PHASE1_EPOCHS  = 15          # Head-only
PHASE2_EPOCHS  = 25          # Unfreeze top-100 backbone layers
PHASE3_EPOCHS  = 20          # Unfreeze top-200 backbone layers
FINE_TUNE_AT1  = 100         # Phase 2: freeze layers[:100]
FINE_TUNE_AT2  = 50          # Phase 3: freeze layers[:50]
SEED           = 42
CLASSES        = ["Happy", "Neutral", "Sad"]
NUM_CLASSES    = len(CLASSES)
SAVE_PATH      = "efficientnetv2s_emotion_best.keras"
LABEL_SMOOTHING = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# 1.  HARDWARE SETUP
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Improved Emotion Detector  v2 — EfficientNetV2S")
print("=" * 60)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓  GPU(s): {[g.name for g in gpus]}")
    mixed_precision.set_global_policy("mixed_float16")
    print("✓  Mixed precision: mixed_float16")
    tf.config.optimizer.set_jit(True)
    print("✓  XLA JIT enabled")
else:
    print("⚠  No GPU detected — training will be slow on CPU")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CUSTOM LOSS  (sparse labels + label smoothing)
# ─────────────────────────────────────────────────────────────────────────────
class SparseLabelSmoothingLoss(keras.losses.Loss):
    """Sparse categorical cross-entropy with label smoothing.

    Works with integer labels so that Keras class_weight still applies.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.smoothing   = smoothing

    def call(self, y_true, y_pred):
        y_true  = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_oh    = tf.one_hot(y_true, self.num_classes)          # (B, C)
        # Smooth: move (smoothing / C) mass to every class
        y_smooth = y_oh * (1.0 - self.smoothing) + self.smoothing / self.num_classes
        # Stable log (cast pred to float32 for numerical safety)
        log_pred = tf.math.log(tf.clip_by_value(
            tf.cast(y_pred, tf.float32), 1e-7, 1.0
        ))
        return tf.reduce_mean(-tf.reduce_sum(y_smooth * log_pred, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_classes": self.num_classes, "smoothing": self.smoothing})
        return cfg

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING & SPLITS
# ─────────────────────────────────────────────────────────────────────────────
all_paths, all_labels = [], []

for label_idx, class_name in enumerate(CLASSES):
    class_dir = DATASET_DIR / class_name
    imgs = (list(class_dir.glob("*.jpg"))  +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png")))
    all_paths.extend([str(p) for p in imgs])
    all_labels.extend([label_idx] * len(imgs))
    print(f"  {class_name:10s} ({label_idx}):  {len(imgs):,} images")

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels, dtype=np.int32)
print(f"\n  Total: {len(all_paths):,} images  |  Classes: {CLASSES}\n")

# 70 / 15 / 15 stratified split
X_temp, X_test, y_temp, y_test = train_test_split(
    all_paths, all_labels, test_size=0.15, random_state=SEED, stratify=all_labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp
)
print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

# Class weights (corrects label imbalance)
cw_values     = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(cw_values))
print(f"\n  Class weights: { {CLASSES[k]: round(v, 3) for k, v in class_weights.items()} }")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  tf.data PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

# Stronger augmentation (images stay in [0, 255])
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.20),
    layers.RandomZoom(0.20),
    layers.RandomTranslation(0.12, 0.12),
    layers.RandomBrightness(0.30),
    layers.RandomContrast(0.30),
], name="augmentation")


def load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    """Decode → resize.  FIX: do NOT normalise — EfficientNetV2S rescales
    internally via include_preprocessing=True, which expects [0, 255]."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)   # keep [0, 255]
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

steps_per_epoch = len(X_train) // BATCH_SIZE
print(f"\n  Batch size: {BATCH_SIZE}  |  Steps/epoch: {steps_per_epoch}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL ARCHITECTURE  (EfficientNetV2S + larger head)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, input_shape=(224, 224, 3)):
    # include_preprocessing=True → model handles [0,255] → internal scale
    base = EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        include_preprocessing=True,
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)

    # Larger classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    # dtype="float32" required with mixed_float16
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return keras.Model(inputs, outputs), base


model, base_model = build_model(NUM_CLASSES)
model.summary(line_length=90)

loss_fn = SparseLabelSmoothingLoss(num_classes=NUM_CLASSES, smoothing=LABEL_SMOOTHING)


def make_callbacks(path):
    return [
        ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=7,
                      restore_best_weights=True, verbose=1),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PHASE 1 — Head-only training with cosine-decay LR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 1: Head-only (backbone frozen)")
print("=" * 60)

lr1 = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=PHASE1_EPOCHS * steps_per_epoch,
    alpha=0.01,
)
model.compile(optimizer=keras.optimizers.Adam(lr1),
              loss=loss_fn, metrics=["accuracy"])

h1 = model.fit(
    train_ds,
    epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=make_callbacks(SAVE_PATH),
    verbose=1,
)
print(f"\n  Phase 1 best val_accuracy: {max(h1.history['val_accuracy']):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PHASE 2 — Unfreeze top-100 backbone layers
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  PHASE 2: Fine-tune backbone layers {FINE_TUNE_AT1}+")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT1]:
    layer.trainable = False

trainable_count = sum(l.trainable for l in base_model.layers)
print(f"  Trainable backbone layers: {trainable_count} / {len(base_model.layers)}")

lr2 = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=PHASE2_EPOCHS * steps_per_epoch,
    alpha=0.01,
)
model.compile(optimizer=keras.optimizers.Adam(lr2),
              loss=loss_fn, metrics=["accuracy"])

h2 = model.fit(
    train_ds,
    epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=make_callbacks(SAVE_PATH),
    verbose=1,
)
print(f"\n  Phase 2 best val_accuracy: {max(h2.history['val_accuracy']):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PHASE 3 — Unfreeze top-200 backbone layers (deep fine-tune)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  PHASE 3: Deep fine-tune (backbone layers {FINE_TUNE_AT2}+)")
print("=" * 60)

for layer in base_model.layers[:FINE_TUNE_AT2]:
    layer.trainable = False
for layer in base_model.layers[FINE_TUNE_AT2:]:
    layer.trainable = True

trainable_count = sum(l.trainable for l in base_model.layers)
print(f"  Trainable backbone layers: {trainable_count} / {len(base_model.layers)}")

lr3 = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-6,
    decay_steps=PHASE3_EPOCHS * steps_per_epoch,
    alpha=0.01,
)
model.compile(optimizer=keras.optimizers.Adam(lr3),
              loss=loss_fn, metrics=["accuracy"])

# Stricter early stopping for deep fine-tune to prevent overfit
h3 = model.fit(
    train_ds,
    epochs=PHASE3_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[
        ModelCheckpoint(SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
    ],
    verbose=1,
)
print(f"\n  Phase 3 best val_accuracy: {max(h3.history['val_accuracy']):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION on held-out test set")
print("=" * 60)

best_model = keras.models.load_model(
    SAVE_PATH,
    custom_objects={"SparseLabelSmoothingLoss": SparseLabelSmoothingLoss},
)
test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
print(f"\n  Test Loss    : {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc * 100:.2f}%\n")

y_pred_proba = best_model.predict(test_ds, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)
y_true       = np.concatenate([y for _, y in test_ds], axis=0)

print(classification_report(y_true, y_pred, target_names=CLASSES))

# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def merge(histories, key):
    result = []
    for h in histories:
        result.extend(h.history[key])
    return result


all_histories = [h1, h2, h3]
p1_end = len(h1.history["accuracy"])
p2_end = p1_end + len(h2.history["accuracy"])
epochs_range = range(1, len(merge(all_histories, "accuracy")) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (train_key, val_key), title in zip(
    axes,
    [("accuracy", "val_accuracy"), ("loss", "val_loss")],
    ["Model Accuracy", "Model Loss"],
):
    ax.plot(epochs_range, merge(all_histories, train_key),
            label=f"Train", linewidth=2)
    ax.plot(epochs_range, merge(all_histories, val_key),
            label=f"Val", linewidth=2)
    ax.axvline(p1_end, color="gray",   linestyle="--", alpha=0.6, label="Phase 2 start")
    ax.axvline(p2_end, color="orange", linestyle="--", alpha=0.6, label="Phase 3 start")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("v2_training_curves.png", dpi=150, bbox_inches="tight")
print("  Saved: v2_training_curves.png")


def plot_confusion_matrix(cm_arr, classes, normalize=True, title="Confusion Matrix"):
    if normalize:
        cm_arr = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks);  ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(tick_marks);  ax.set_yticklabels(classes, fontsize=12,
                                                    rotation=90, va="center")
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
fig_cm = plot_confusion_matrix(
    cm_arr, CLASSES, normalize=True,
    title=f"v2 Emotion Classifier — Test Acc {test_acc * 100:.1f}%",
)
fig_cm.savefig("v2_emotion_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("  Saved: v2_emotion_confusion_matrix.png")

plt.show()
print(f"\n✓  Training complete.  Best model saved to: {SAVE_PATH}")
