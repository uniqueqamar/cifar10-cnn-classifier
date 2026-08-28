# Image Classification for Common Objects

A CNN trained from scratch (TensorFlow/Keras) to classify everyday objects into 10 categories using the CIFAR-10 dataset.

## Problem

Assign one of 10 labels — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck — to a 32×32 color image.

## Approach

- **Data**: CIFAR-10 (60,000 32×32 color images, 10 balanced classes). Split into 45,000 train / 5,000 validation / 10,000 test — the test set is only touched once, for final evaluation.
- **Augmentation**: random rotation, width/height shift, horizontal flip, and zoom applied to training data only, to reduce overfitting on such small images.
- **Architecture**: 3 convolutional blocks (32 → 64 → 128 filters) with BatchNormalization and MaxPooling after each, followed by a dense classification head with Dropout.
- **Training**: Adam optimizer, categorical cross-entropy loss, with `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks so training stops and saves automatically at the best validation loss rather than a fixed epoch count.

## Results

| Metric | Value |
|---|---|
| Test accuracy | _fill in after running_ |
| Best val loss | _fill in after running_ |

See `outputs/training_curves.png` for accuracy/loss curves and `outputs/confusion_matrix.png` for the per-class breakdown after training — the classes most often confused for each other on CIFAR-10 are typically cat/dog and automobile/truck, worth calling out once you have your own numbers.

## Project structure

```
.
├── train.py                       # training script (run this)
├── requirements.txt
└── outputs/                       # created after running train.py
    ├── model.keras                # best saved model
    ├── training_curves.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

## How to run

```bash
pip install -r requirements.txt
python train.py
```

Training runs for up to 60 epochs but stops early once validation loss stops improving (typically 20–35 epochs on a GPU, longer on CPU).

## Possible extensions

- Swap the from-scratch CNN for a pretrained backbone (e.g. `MobileNetV2` or `ResNet50` via transfer learning) — usually pushes CIFAR-10 test accuracy well past 90% with far less training.
- Add a small Streamlit or Gradio app for uploading a custom image and getting a live prediction.
- Try CIFAR-100 or a custom-scraped dataset to move past the standard benchmark.
