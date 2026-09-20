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
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.models import Sequential
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "images"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
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
    """Load images, shuffle once with a fixed seed, split at the image level,
    then batch. Splitting before batching (with a non-reshuffling shuffle)
    guarantees train/val/test never overlap."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),
        image_size=IMAGE_SIZE,
        batch_size=None,  # unbatched, so we can split per image
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
        ds = ds.map(
            lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.prefetch(tf.data.AUTOTUNE)

    train = prepare(dataset.take(train_size))
    val = prepare(dataset.skip(train_size).take(val_size))
    test = prepare(dataset.skip(train_size + val_size).take(test_size))

    print(f"Total images: {total} | train: {train_size}, val: {val_size}, test: {test_size}")
    return train, val, test, class_names


def build_model() -> Sequential:
    """CNN for binary image classification."""
    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(),
        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
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
    """Evaluate the trained model on the held-out test set."""
    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()

    for images, labels in test_dataset.as_numpy_iterator():
        predictions = model.predict(images, verbose=0).reshape(-1)
        labels = labels.reshape(-1)

        precision.update_state(labels, predictions)
        recall.update_state(labels, predictions)
        accuracy.update_state(labels, predictions)

    print(
        f"Precision: {precision.result().numpy():.4f}, "
        f"Recall: {recall.result().numpy():.4f}, "
        f"Accuracy: {accuracy.result().numpy():.4f}"
    )


def predict_image(model, image_path: str, class_names=None) -> float:
    """Predict a single image. Returns the probability of class index 1."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE).astype("float32") / 255.0

    prediction = float(model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0])

    # Keras assigns class indices alphabetically by folder name
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

    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR))
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        callbacks=[tensorboard_callback],
        verbose=2,  # one plain-text line per epoch, no Unicode progress bar
    )

    plot_training_history(history)
    evaluate_model(model, test_data)

    model_path = MODEL_DIR / "imageclassifier.keras"
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")

    # Example single-image prediction
    sample_image = DATA_DIR / "happy" / "3.1-hero-jtube-mob_tcm1453-112262.jpg"
    if sample_image.exists():
        predict_image(model, str(sample_image), class_names)


if __name__ == "__main__":
    main()