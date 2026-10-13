# Emotion Image Classifier (Angry / Happy / Sad)

A small image classifier built with TensorFlow/Keras that predicts the emotion shown in a picture: **angry**, **happy** or **sad**. The repository contains a from-scratch CNN pipeline, a MobileNetV2 transfer-learning script, and a single-image prediction script, all trained on a small hand-collected dataset of 210 images.

![Dataset samples](dataset_contact_sheet.jpg)

## Features

- Automatic dataset cleanup (removes corrupted and unsupported image files)
- Contact sheet of sample images from each class (`dataset_contact_sheet.jpg`)
- 70 / 20 / 10 train / validation / test split with a fixed seed
- Data augmentation (random horizontal flip and rotation)
- Training curves saved as `training_performance.png`
- Per-class precision, recall and F1 on the test set
- TensorBoard logging
- Command-line prediction for a single image (`predict.py`)

## Project structure

```
emotion-image-classifier/
├── image_classifier.py        # 3-class custom CNN: train, evaluate, save model
├── main.py                    # MobileNetV2 transfer-learning script (see Known issues)
├── predict.py                 # predict the emotion of a single image
├── requirements.txt
├── dataset_contact_sheet.jpg  # sample images per class (generated)
├── training_performance.png   # loss/accuracy curves (generated)
├── images/
│   ├── angry/                 # 70 images
│   ├── happy/                 # 71 images
│   └── sad/                   # 69 images
├── models/                    # saved model (created on first run)
└── logs/                      # TensorBoard logs (created on first run)
```

## Setup

Tested with Python 3.10 and TensorFlow 2.18 on Windows (CPU only).

```bash
git clone https://github.com/Harshh00100/emotion-image-classifier.git
cd emotion-image-classifier

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

> `image_classifier.py` imports `imghdr`, which was removed in Python 3.13. Use Python 3.10 to 3.12.

## Dataset

Images live in one folder per class. Class indices follow alphabetical folder order: `angry = 0`, `happy = 1`, `sad = 2`.

```
images/
├── angry/   # 70 images
├── happy/   # 71 images
└── sad/     # 69 images
```

**Warning:** both training scripts permanently **delete** files in `images/` that are corrupted or not valid images. Keep a backup of your data before running them.

Notes about the data:

- The dataset is very small (210 images), so the model can overfit and the test set is only a few dozen images.
- The images in `angry/` are 48 x 48 pixels, while most `happy/` and `sad/` images are larger web photos. The model may partly learn image source or resolution instead of facial expression, so treat the results as a rough demo, not a benchmark.

## Usage

### 1. Train the custom CNN (`image_classifier.py`)

```bash
python image_classifier.py
```

The script will:

1. Remove invalid images from `images/`
2. Save a sample grid to `dataset_contact_sheet.jpg`
3. Load the data, scale pixels to [0, 1] and split it 70 / 20 / 10 into train / validation / test
4. Train for 20 epochs (batch size 32, seed 42) with TensorBoard logging
5. Save the loss/accuracy curves to `training_performance.png`
6. Print overall accuracy and per-class precision, recall and F1 on the test set, plus one sample prediction
7. Save the model to `models/imageclassifier.keras`

View the training logs with:

```bash
tensorboard --logdir logs
```

### 2. Predict on a new image (`predict.py`)

Run after training with `image_classifier.py` so that `models/imageclassifier.keras` exists:

```bash
python predict.py path/to/image.jpg
```

Example output:

```
===================================
File:       path/to/image.jpg
Prediction: HAPPY (87.12%)
-----------------------------------
 - angry  :   5.30%
 - happy  :  87.12%
 - sad    :   7.58%
===================================
```

If you run `python predict.py` without an argument, it predicts the first image in `images/happy/`. The class names (`angry`, `happy`, `sad`) are hard-coded in `predict.py`, so update `CLASS_NAMES` if you change the dataset folders.

### 3. Transfer learning with MobileNetV2 (`main.py`)

```bash
python main.py
```

This script cleans the data, splits it 70 / 20 / 10 per image, and trains a frozen MobileNetV2 backbone with flip/rotation augmentation, dropout, early stopping and best-model checkpointing. See **Known issues** below before relying on it.

## Model architecture

### Custom CNN (`image_classifier.py`)

| Layer | Details |
|---|---|
| Input | 256 x 256 x 3, scaled to [0, 1] |
| Augmentation | RandomFlip (horizontal), RandomRotation (0.1) |
| Conv2D + MaxPooling | 16 filters, 3x3, ReLU |
| Conv2D + MaxPooling | 32 filters, 3x3, ReLU |
| Conv2D + MaxPooling | 16 filters, 3x3, ReLU |
| GlobalAveragePooling2D | |
| Dense | 256 units, ReLU |
| Dense (output) | 3 units, softmax |

Loss: categorical cross-entropy. Optimizer: Adam. Epochs: 20.

### MobileNetV2 transfer learning (`main.py`)

| Layer | Details |
|---|---|
| Input | 256 x 256 x 3 (raw pixels, 0-255) |
| Augmentation | RandomFlip (horizontal), RandomRotation (0.1) |
| Preprocessing | `mobilenet_v2.preprocess_input` |
| Backbone | MobileNetV2, ImageNet weights, frozen |
| Head | GlobalAveragePooling2D, Dropout (0.3), Dense (1, sigmoid) |

Optimizer: Adam (learning rate 1e-3). Up to 30 epochs with early stopping on validation loss (patience 5).

## Results

Training curves from the custom CNN (`image_classifier.py`, 20 epochs):

![Training curves](training_performance.png)

By epoch 20, training accuracy is about 70% and validation accuracy about 80%, with the loss still decreasing, so the model has not fully converged. Exact test-set precision, recall and F1 are printed at the end of each run and are not stored in the repository.

Because the dataset is tiny and the split is coarse, these numbers are noisy and should be read as a sanity check, not a reliable benchmark.

## Known issues

These are documented here until they are fixed:

- `main.py` uses a single sigmoid output and binary cross-entropy, which does not match the three-class dataset, so its training, evaluation and prediction logic need updating.
- `main.py` and `image_classifier.py` both write to `models/imageclassifier.keras` and expect different input scaling, and `predict.py` matches only the `image_classifier.py` model.
- `image_classifier.py` splits the data by batch rather than by image, so the split sizes are approximate.

## Ideas for improvement

- Fix the issues above and merge the two training scripts into one pipeline
- Use a stratified split and k-fold cross-validation for a more reliable estimate
- Add a confusion matrix and Grad-CAM visualisations
- Use consistent, properly licensed data for all three classes

## Tech stack

Python, TensorFlow / Keras, OpenCV, Pillow, NumPy, Matplotlib
