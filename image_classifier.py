import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras import callbacks, layers, models

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "images"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
MODEL_PATH = MODELS_DIR / "imageclassifier.keras"
CONTACT_SHEET_PATH = ROOT_DIR / "dataset_contact_sheet.jpg"
TRAINING_PLOT_PATH = ROOT_DIR / "training_performance.png"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_images(data_dir: Path, delete_invalid: bool = False) -> list[Path]:
    """Report invalid image files; delete them only when explicitly requested."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    invalid_files = []
    checked = 0
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            # Ignore metadata, hidden files, and unsupported extensions rather than
            # treating them as images (for example, .gitkeep).
            if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            checked += 1
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                invalid_files.append(path)
                print(f"Invalid image: {path} ({exc})")
                if delete_invalid:
                    path.unlink()
                    print(f"Deleted by explicit request: {path}")

    print(f"Image validation complete: checked {checked}; invalid {len(invalid_files)}.")
    if invalid_files and not delete_invalid:
        print("No files were deleted. Review the list above, repair/move files, or rerun with --delete-invalid.")
    return invalid_files


def build_datasets():
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),
        image_size=IMAGE_SIZE,
        batch_size=None,
        label_mode="int",
        color_mode="rgb",
        shuffle=False,
        follow_links=False,
    )
    class_names = dataset.class_names
    total = int(tf.data.experimental.cardinality(dataset).numpy())
    if total < 3:
        raise ValueError(f"At least 3 valid images are required; found {total}.")

    dataset = dataset.shuffle(total, seed=SEED, reshuffle_each_iteration=False)
    train_size = int(total * 0.7)
    val_size = int(total * 0.2)
    test_size = total - train_size - val_size

    def prepare(ds):
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train = prepare(dataset.take(train_size))
    val = prepare(dataset.skip(train_size).take(val_size))
    test = prepare(dataset.skip(train_size + val_size).take(test_size))
    print(f"Images: {total} | train: {train_size}, validation: {val_size}, test: {test_size}")
    print(f"Class order: {class_names}")
    return train, val, test, class_names


def build_model(num_classes: int):
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.Rescaling(1.0 / 255.0),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.Conv2D(16, 3, activation="relu", name="conv1"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu", name="conv2"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, activation="relu", name="last_conv"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def generate_contact_sheet(data_dir: Path, output_path: Path, samples_per_class: int = 4):
    class_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        print("No class folders found; skipping contact sheet.")
        return
    fig, axes = plt.subplots(len(class_dirs), samples_per_class,
                             figsize=(samples_per_class * 3, len(class_dirs) * 3),
                             squeeze=False)
    for row, class_dir in enumerate(class_dirs):
        files = [p for p in sorted(class_dir.iterdir())
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        for col in range(samples_per_class):
            axes[row, col].axis("off")
            if col >= len(files):
                continue
            try:
                with Image.open(files[col]) as image:
                    axes[row, col].imshow(image.convert("RGB"))
                axes[row, col].set_title(class_dir.name)
            except (OSError, ValueError):
                axes[row, col].set_title("Unreadable")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Contact sheet saved to: {output_path}")


def plot_training_history(history, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        axis.plot(history.history.get(metric, []), label=f"train_{metric}")
        axis.plot(history.history.get(f"val_{metric}", []), label=f"val_{metric}")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Training plot saved to: {output_path}")


def evaluate_model(model, test_data, class_names):
    y_true, y_pred = [], []
    for images, labels in test_data:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())
    if not y_true:
        print("Test split is empty; no evaluation metrics available.")
        return
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    print(f"Test accuracy: {np.mean(y_true == y_pred):.4f}")
    print("Per-class metrics:")
    for idx, name in enumerate(class_names):
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"  {name}: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train the custom emotion CNN.")
    parser.add_argument("--delete-invalid", action="store_true",
                        help="Permanently delete invalid supported image files; default is report-only.")
    args = parser.parse_args()

    invalid = validate_images(DATA_DIR, delete_invalid=args.delete_invalid)
    if invalid and not args.delete_invalid:
        raise SystemExit("Training stopped to protect the dataset. Fix/move the listed files or rerun with --delete-invalid.")
    generate_contact_sheet(DATA_DIR, CONTACT_SHEET_PATH)
    train_data, val_data, test_data, class_names = build_datasets()
    model = build_model(len(class_names))
    model.summary()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    training_callbacks = [
        callbacks.TensorBoard(log_dir=str(LOGS_DIR)),
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS,
                        callbacks=training_callbacks, verbose=2)
    plot_training_history(history, TRAINING_PLOT_PATH)
    evaluate_model(model, test_data, class_names)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
