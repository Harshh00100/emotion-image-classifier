# -*- coding: utf-8 -*-

import sys
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
import os

# Set before importing TensorFlow so the settings take effect
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import (Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input,
                          MaxPooling2D, RandomContrast, RandomFlip, RandomRotation,
                          RandomZoom, BatchNormalization)
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.models import Sequential
from keras import regularizers
from PIL import Image
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "images"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 30  # EarlyStopping will automatically stop training when optimal
SEED = 42
VALID_FORMATS = {"JPEG", "PNG", "BMP"}


def clean_invalid_images(data_dir: Path) -> None:
    """Remove corrupted files and files that are not JPEG/PNG/BMP images."""
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
                    img.verify()  # detects truncated/corrupted files
                if img_format not in VALID_FORMATS:
                    raise ValueError(f"Unsupported format: {img_format}")
            except Exception as exc:
                print(f"Removing invalid image: {file_path} ({exc})")
                try:
                    file_path.unlink()
                except Exception as unlink_exc:
                    print(f"Could not delete {file_path}: {unlink_exc}")


def build_datasets():
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),
        image_size=IMAGE_SIZE,
        batch_size=None,
        color_mode="rgb",
        shuffle=False,
    )
    class_names = getattr(dataset, "class_names", None)

    total = int(tf.data.experimental.cardinality(dataset).numpy())
    dataset = dataset.shuffle(total, seed=SEED, reshuffle_each_iteration=False)

    train_size = int(total * 0.7)
    val_size = int(total * 0.2)
    test_size = total - train_size - val_size

    def prepare(ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.batch(BATCH_SIZE)
        return ds.prefetch(tf.data.AUTOTUNE)

    train = prepare(dataset.take(train_size))
    val = prepare(dataset.skip(train_size).take(val_size))
    test = prepare(dataset.skip(train_size + val_size).take(test_size))

    print(f"Total images: {total} | train: {train_size}, val: {val_size}, test: {test_size}")
    return train, val, test, class_names


def build_model() -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pre-trained feature extractor

    inputs = Input(shape=(*IMAGE_SIZE, 3))

    # Augmentation pipeline
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)

    # MobileNetV2 preprocessing expects inputs in range [-1, 1]
    x = preprocess_input(x)

    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history) -> None:
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["loss"], color="teal", label="train_loss")
    axes[0].plot(history.history["val_loss"], color="orange", label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], color="teal", label="train_accuracy")
    axes[1].plot(history.history["val_accuracy"], color="orange", label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_dataset: tf.data.Dataset) -> None:
    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0).reshape(-1)
        labels_flat = tf.cast(labels, tf.float32).numpy().reshape(-1)

        precision.update_state(labels_flat, predictions)
        recall.update_state(labels_flat, predictions)
        accuracy.update_state(labels_flat, predictions)

    print(
        f"Precision: {precision.result().numpy():.4f}, "
        f"Recall: {recall.result().numpy():.4f}, "
        f"Accuracy: {accuracy.result().numpy():.4f}"
    )


def predict_image(model, image_path: str, class_names=None) -> float:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE).astype("float32")

    prediction = float(model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0])

    names = class_names or ["happy", "sad"]
    label = names[1] if prediction > 0.5 else names[0]
    print(f"Image: {image_path} -> {label} ({prediction:.4f})")
    return prediction


def main() -> None:
    clean_invalid_images(DATA_DIR)

    train_data, val_data, test_data, class_names = build_datasets()
    print(f"Class names (index order): {class_names}")

    model = build_model()
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "imageclassifier.keras"

    # Callbacks to prevent overfitting and save best model weights
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR)),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_history(history)
    evaluate_model(model, test_data)

    print(f"Best model saved to: {model_path}")

    # Example single-image prediction
    if class_names and len(class_names) > 0:
        sample_folder = DATA_DIR / class_names[0]
        sample_files = list(sample_folder.glob("*.*"))
        if sample_files:
            predict_image(model, str(sample_files[0]), class_names)


if __name__ == "__main__":
    main()