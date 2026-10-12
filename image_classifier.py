
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import (
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
)
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.models import Sequential
from PIL import Image


# Configuration
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "images"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42
VALID_FORMATS = {"JPEG", "PNG", "BMP"}

tf.keras.utils.set_random_seed(SEED)


# Validate images
def clean_invalid_images(data_dir: Path) -> None:
    """Remove corrupted images and unsupported formats."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for file_path in class_dir.iterdir():
            if not file_path.is_file():
                continue

            try:
                with Image.open(file_path) as img:
                    img_format = img.format
                    img.verify()

                if img_format not in VALID_FORMATS:
                    raise ValueError(f"Unsupported format: {img_format}")

            except Exception as exc:
                print(f"Invalid image: {file_path} ({exc})")
                # Leave the original file in place for manual inspection.
                # This avoids permanently deleting dataset files.


# Build stratified datasets
def build_datasets():
    """Create reproducible, class-balanced train/validation/test splits."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),
        image_size=IMAGE_SIZE,
        batch_size=None,
        color_mode="rgb",
        shuffle=False,
    )

    class_names = dataset.class_names

    if len(class_names) != 2:
        raise ValueError(
            f"Expected exactly 2 classes, found {len(class_names)}: "
            f"{class_names}"
        )

    images, labels = [], []

    for image, label in dataset.as_numpy_iterator():
        images.append(image)
        labels.append(int(label))

    images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.int32)

    total = len(labels)
    if total < 10:
        raise ValueError(f"Dataset too small: {total} images")

    rng = np.random.default_rng(SEED)
    train_indices, val_indices, test_indices = [], [], []

    for class_id, class_name in enumerate(class_names):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)

        n = len(class_indices)
        if n < 3:
            raise ValueError(
                f"Class '{class_name}' has only {n} images."
            )

        n_train = max(1, min(int(n * 0.7), n - 2))
        n_val = max(1, min(int(n * 0.2), n - n_train - 1))

        train_indices.extend(class_indices[:n_train].tolist())
        val_indices.extend(
            class_indices[n_train:n_train + n_val].tolist()
        )
        test_indices.extend(
            class_indices[n_train + n_val:].tolist()
        )

    def prepare(indices, shuffle=False):
        indices = np.asarray(indices, dtype=np.int64)

        if shuffle:
            rng.shuffle(indices)

        ds = tf.data.Dataset.from_tensor_slices(
            (images[indices], labels[indices])
        )

        ds = ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(BATCH_SIZE)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_data = prepare(train_indices, shuffle=True)
    val_data = prepare(val_indices)
    test_data = prepare(test_indices)

    print(
        f"Total: {total} | Train: {len(train_indices)} | "
        f"Validation: {len(val_indices)} | Test: {len(test_indices)}"
    )

    for name, indices in [
        ("Train", train_indices),
        ("Validation", val_indices),
        ("Test", test_indices),
    ]:
        counts = np.bincount(
            labels[np.asarray(indices, dtype=np.int64)],
            minlength=len(class_names),
        )
        print(f"{name} class counts: {dict(zip(class_names, counts.tolist()))}")

    return train_data, val_data, test_data, class_names


# Build compact CNN
def build_model() -> Sequential:
    """Build a compact CNN with training-time data augmentation."""
    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),

        # Active during training; inactive during validation and prediction.
        RandomFlip("horizontal"),
        RandomRotation(0.05),

        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D(),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(),

        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D(),

        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


# Plot training history
def plot_training_history(history) -> None:
    """Plot training and validation loss and accuracy."""
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["loss"], label="Training loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Training accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# Evaluate model
def evaluate_model(model, test_dataset: tf.data.Dataset) -> None:
    """Report precision, recall, and accuracy for the test set."""
    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0).reshape(-1)
        labels = tf.reshape(labels, [-1])

        precision.update_state(labels, predictions)
        recall.update_state(labels, predictions)
        accuracy.update_state(labels, predictions)

    print("\nTest Results")
    print("-" * 35)
    print(f"Precision: {precision.result().numpy():.4f}")
    print(f"Recall:    {recall.result().numpy():.4f}")
    print(f"Accuracy:  {accuracy.result().numpy():.4f}")


# Predict one image
def predict_image(model, image_path: str, class_names=None) -> float:
    """Predict an image and return the probability of class index 1."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Image not found or cannot be read: {image_path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0

    probability = float(
        model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
    )

    names = class_names or ["happy", "sad"]
    if len(names) != 2:
        raise ValueError("Exactly two class names are required.")

    predicted_class = names[1] if probability >= 0.5 else names[0]

    print(
        f"Image: {image_path}\n"
        f"Prediction: {predicted_class}\n"
        f"Class-1 probability: {probability:.4f}"
    )

    return probability


# Main training pipeline
def main() -> None:
    clean_invalid_images(DATA_DIR)

    train_data, val_data, test_data, class_names = build_datasets()
    print(f"Class names (index order): {class_names}")

    model = build_model()
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR / "training")
        ),
    ]

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=2,
        shuffle=False,
    )

    plot_training_history(history)
    evaluate_model(model, test_data)

    model_path = MODEL_DIR / "imageclassifier.keras"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    sample_image = (
        DATA_DIR / "happy" / "3.1-hero-jtube-mob_tcm1453-112262.jpg"
    )

    if sample_image.exists():
        predict_image(model, str(sample_image), class_names)


if __name__ == "__main__":
    main()
