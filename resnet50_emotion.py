"""
ResNet50 Transfer Learning for Emotion Detection
=================================================
Paste each section as a new cell in your notebook after the train/test split.
Assumes X_train, X_test, ytrain, ytest are already defined.
"""

# ============================================================
# CELL 1 — Resize images to 224x224 (ResNet50 optimal size)
# ============================================================
import cv2
import numpy as np

TARGET_SIZE = (224, 224)

# Normalize to [0, 1] (images already loaded at 224x224)
X_train = X_train / 255.0
X_test  = X_test  / 255.0

print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")


# ============================================================
# CELL 2 — Build ResNet50 model
# ============================================================
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_resnet50(input_shape=(224, 224, 3), num_classes=3):
    # Load ResNet50 with ImageNet weights, remove the top classifier
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Phase 1: freeze the entire base — only train the custom head
    base_model.trainable = False

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Return base_model so Phase 2 can unfreeze specific layers
    return model, base_model

resnet_model, resnet_base = build_resnet50(input_shape=(224, 224, 3), num_classes=3)
resnet_model.summary()


# ============================================================
# CELL 3 — Phase 1: Train head only (fast, ~15 epochs)
# ============================================================
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("Phase 1: Training classification head (base frozen)...")
history_phase1 = resnet_model.fit(
    X_train, ytrain,
    epochs=15,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1,
    callbacks=callbacks
)


# ============================================================
# CELL 4 — Phase 2: Fine-tune last 30 layers of ResNet50
# ============================================================
print("\nPhase 2: Fine-tuning last 30 layers of ResNet50...")

# Unfreeze the whole base, then refreeze everything except last 30 layers
resnet_base.trainable = True
for layer in resnet_base.layers[:-30]:
    layer.trainable = False

# Recompile with very small learning rate to avoid destroying pre-trained weights
resnet_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_phase2 = resnet_model.fit(
    X_train, ytrain,
    epochs=30,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1,
    callbacks=callbacks
)


# ============================================================
# CELL 5 — Evaluate & Save
# ============================================================
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

pred_resnet = np.argmax(resnet_model.predict(X_test), axis=1)

# Reuse your existing plot_confusion_matrix function
plot_confusion_matrix(
    confusion_matrix(ytest, pred_resnet),
    classes=['Happy', 'Neutral', 'Sad'],
    normalize=True,
    title='ResNet50 Confusion Matrix'
)
plt.show()

test_loss, test_acc = resnet_model.evaluate(X_test, ytest, verbose=0)
print(f"ResNet50 Test Accuracy: {test_acc * 100:.2f}%")

resnet_model.save('resnet50_emotion.keras')
print("Saved: resnet50_emotion.keras")
