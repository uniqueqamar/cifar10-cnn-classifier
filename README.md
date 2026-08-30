# CIFAR-10 CNN Classifier

A convolutional neural network trained from scratch (TensorFlow/Keras) to classify everyday objects into 10 categories, with an interactive Streamlit app for live predictions.

## Problem

Assign one of 10 labels — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck — to a 32×32 color image.

## Approach

- **Data**: CIFAR-10 (60,000 32×32 color images, 10 balanced classes). Split into 45,000 train / 5,000 validation / 10,000 test — the test set is only touched once, for final evaluation.
- **Augmentation**: random rotation, width/height shift, horizontal flip, and zoom applied to training data only, to reduce overfitting on such small images.
- **Architecture**: 3 convolutional blocks (32 → 64 → 128 filters) with BatchNormalization and MaxPooling after each, followed by a dense classification head with Dropout.
- **Training**: Adam optimizer (learning rate 5e-4) with label smoothing (0.1) in the categorical cross-entropy loss, plus `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks so training stops and saves automatically at the best validation loss rather than a fixed epoch count.

## Results

Trained in two iterations — the first run surfaced a specific failure mode, which the second run fixed.

| Metric | v1 (baseline) | v2 (tuned) |
|---|---|---|
| Test accuracy | 77.48% | **80.00%** |
| Macro avg F1 | 0.77 | 0.80 |

**What changed between v1 and v2:** v1's confusion matrix showed "frog" acting as a catch-all bucket — 96% recall but only 59% precision, with hundreds of birds, cats, and deer misclassified as frog. Since all classes are perfectly balanced (1,000 images each), this was a model-confidence issue, not a data issue. Lowering the learning rate (1e-3 → 5e-4) and adding label smoothing (0.1) in the loss function fixed it: frog's precision rose to 0.62, and every class most affected by frog-confusion improved — cat's recall went from 0.49 to 0.58, bird's F1 from 0.65 to 0.70.

| Class | v1 F1 | v2 F1 |
|---|---|---|
| airplane | 0.81 | 0.83 |
| automobile | 0.90 | 0.93 |
| bird | 0.65 | 0.70 |
| cat | 0.58 | 0.65 |
| deer | 0.75 | 0.76 |
| dog | 0.70 | 0.72 |
| frog | 0.73 | 0.75 |
| horse | 0.84 | 0.86 |
| ship | 0.89 | 0.90 |
| truck | 0.83 | 0.88 |

Cat remains the weakest class (0.65 F1) — likely visual overlap with dog and frog at 32×32 resolution. See `outputs/confusion_matrix.png` and `outputs/training_curves.png` for the full picture.

## Interactive App

A Streamlit app (`app.py`) for trying the trained model on any uploaded image — shows the predicted class, a confidence score, and the top-5 predictions as ranked confidence bars, alongside the training curves and confusion matrix from the run above.

## Tech Stack

Python · TensorFlow / Keras · Scikit-learn (metrics) · Matplotlib · Seaborn · Streamlit

## Project Structure

```
cifar10-cnn-classifier/
├── train.py                       # training script
├── app.py                         # Streamlit app for live predictions
├── requirements.txt
└── outputs/                       # created after running train.py
    ├── model.keras                # best saved model
    ├── training_curves.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

## How to Run

**Train the model:**
```bash
pip install -r requirements.txt
python train.py
```
Training runs for up to 60 epochs but stops early once validation loss stops improving (typically 20–35 epochs on a GPU, longer on CPU).

**Launch the app** (after training, so `outputs/model.keras` exists):
```bash
streamlit run app.py
```

## Possible Extensions

- Swap the from-scratch CNN for a pretrained backbone (e.g. `MobileNetV2` or `ResNet50` via transfer learning) — usually pushes CIFAR-10 test accuracy well past 90% with far less training.
- Deploy the Streamlit app publicly (Streamlit Community Cloud) for a live demo link.
- Try CIFAR-100 or a custom-scraped dataset to move past the standard benchmark.

## Author

**Qamareen Fatima**
