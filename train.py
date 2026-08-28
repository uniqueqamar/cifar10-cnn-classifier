"""
CIFAR-10 image classifier — CNN trained from scratch with TensorFlow/Keras.

Run: python train.py
Outputs (written to ./outputs/):
    model.keras              - best model weights (by val_loss)
    training_curves.png      - accuracy & loss over epochs
    confusion_matrix.png     - per-class confusion matrix on the test set
    classification_report.txt - precision/recall/F1 per class
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# ---------------------------------------------------------------------------
# 1. Load data and split properly: train / val / test (test is touched once)
# ---------------------------------------------------------------------------
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

VAL_SIZE = 5000
x_val, y_val = x_train_full[-VAL_SIZE:], y_train_full[-VAL_SIZE:]
x_train, y_train = x_train_full[:-VAL_SIZE], y_train_full[:-VAL_SIZE]

y_train_cat = to_categorical(y_train, 10)
y_val_cat = to_categorical(y_val, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")

# ---------------------------------------------------------------------------
# 2. Data augmentation — only applied to training data
# ---------------------------------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
datagen.fit(x_train)

# ---------------------------------------------------------------------------
# 3. Model — deeper, with BatchNorm
# ---------------------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "model.keras"),
                     monitor='val_loss', save_best_only=True),
]

# ---------------------------------------------------------------------------
# 5. Train — validation is the held-out val set, never the test set
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 60  # EarlyStopping will typically stop well before this

print("Training the model...")
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val_cat),
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 6. Final evaluation — test set touched exactly once, here
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"\nFinal test accuracy: {test_acc*100:.2f}%")

# ---------------------------------------------------------------------------
# 7. Save training curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'], label='train')
axes[0].plot(history.history['val_accuracy'], label='val')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'], label='train')
axes[1].plot(history.history['val_loss'], label='val')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
plt.close()

# ---------------------------------------------------------------------------
# 8. Confusion matrix + classification report on the test set
# ---------------------------------------------------------------------------
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = y_test.flatten()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Test Accuracy: {test_acc*100:.2f}%)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
    f.write(report)

print(report)
<<<<<<< HEAD
print(f"\nAll artifacts saved to ./{OUTPUT_DIR}/")
=======
print(f"\nAll artifacts saved to ./{OUTPUT_DIR}/")
>>>>>>> 89e797671ef1c9c3ba2d82e975721144d047c8dc
