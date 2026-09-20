# Happy vs Sad Image Classifier (CNN)

A binary image classifier built from scratch with TensorFlow/Keras. A small convolutional neural network is trained to tell **happy** images from **sad** images, using a hand-collected dataset of 140 images.

## Features

- Automatic dataset cleanup (removes corrupted files and unsupported formats)
- Leak-free train / validation / test split (70% / 20% / 10%): images are shuffled once with a fixed seed and split before batching, so the sets never overlap
- CNN with three convolution + max-pooling blocks and a sigmoid output
- Training curves (loss and accuracy) plotted after training
- Test-set evaluation with precision, recall and accuracy
- TensorBoard logging
- Single-image prediction helper

## Project structure

```
image_classifier/
├── image_classifier.py    # training, evaluation and prediction script
├── requirements.txt
├── images/
│   ├── happy/             # 71 images
│   └── sad/               # 69 images
├── models/                # saved model (created on first run)
└── logs/                  # TensorBoard logs (created on first run)
```

## Model architecture

| Layer | Output shape |
|---|---|
| Input | 256 x 256 x 3 |
| Conv2D (16, 3x3, ReLU) + MaxPooling | 127 x 127 x 16 |
| Conv2D (32, 3x3, ReLU) + MaxPooling | 62 x 62 x 32 |
| Conv2D (16, 3x3, ReLU) + MaxPooling | 30 x 30 x 16 |
| Flatten | 14400 |
| Dense (256, ReLU) | 256 |
| Dense (1, Sigmoid) | 1 |

About 3.7M trainable parameters. Loss is binary cross-entropy, optimizer is Adam, and pixel values are normalized to [0, 1].

## Setup

Tested with Python 3.10 and TensorFlow 2.18 on Windows (CPU only).

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Usage

Organize your images into one folder per class:

```
images/
├── happy/
└── sad/
```

Then run:

```bash
python image_classifier.py
```

The script will:

1. Clean invalid images from `images/`
2. Build and split the dataset
3. Train for 20 epochs and plot the loss/accuracy curves
4. Print precision, recall and accuracy on the test set
5. Save the model to `models/imageclassifier.keras`

To view training logs in TensorBoard:

```bash
tensorboard --logdir logs
```

### Predicting on a new image

```python
from keras.models import load_model
from image_classifier import predict_image

model = load_model("models/imageclassifier.keras")
predict_image(model, "path/to/image.jpg", ["happy", "sad"])
```

Class indices follow alphabetical folder order: `happy = 0`, `sad = 1`. A probability above 0.5 means **sad**.

## Results

_Add your results here after training, for example:_

| Metric | Value |
|---|---|
| Precision | - |
| Recall | - |
| Accuracy | - |

Note: the test split contains only about 14 images, so these numbers are noisy and should be read as a rough sanity check, not a reliable benchmark.

## Limitations and ideas for improvement

- Small dataset (140 images), so the model can overfit
- Add data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`)
- Add `Dropout` and early stopping
- Try transfer learning (for example MobileNetV2 or EfficientNet) for better accuracy with little data
- Use k-fold cross-validation for a more reliable estimate

## Tech stack

Python, TensorFlow / Keras, OpenCV, Pillow, NumPy, Matplotlib

## License

Add a license of your choice (for example MIT) if you want others to reuse the code. Check the licensing of the images before redistributing the dataset.
