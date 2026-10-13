import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (256, 256)
DEFAULT_MODEL = ROOT_DIR / "models" / "imageclassifier.keras"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def class_names_for(model_path: Path) -> list[str]:
    # Directory-loader class order is alphabetical.
    data_dir = ROOT_DIR / "images"
    if data_dir.is_dir():
        names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
        if names:
            return names
    return ["angry", "happy", "sad"]


def predict_image(image_path: Path, model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train image_classifier.py or main.py first."
        )
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
        pixels = np.asarray(image, dtype=np.float32)

    # Both models accept raw RGB pixels in [0, 255]; preprocessing is inside each model.
    model = tf.keras.models.load_model(model_path)
    probabilities = np.asarray(model.predict(np.expand_dims(pixels, axis=0), verbose=0)[0])
    names = class_names_for(model_path)
    if probabilities.ndim != 1 or len(probabilities) != len(names):
        raise ValueError(
            f"Model returned {len(probabilities)} outputs but {len(names)} class names were found. "
            "Ensure the model and images/ class folders match."
        )

    predicted_idx = int(np.argmax(probabilities))
    print(f"File: {image_path}")
    print(f"Model: {model_path.name}")
    print(f"Prediction: {names[predicted_idx].upper()} ({probabilities[predicted_idx] * 100:.2f}%)")
    for name, probability in zip(names, probabilities):
        print(f"  {name:<8}: {probability * 100:6.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Predict the emotion in an image.")
    parser.add_argument("image", nargs="?", type=Path, help="Path to an input image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Model path (default: models/imageclassifier.keras).")
    args = parser.parse_args()
    if args.image is None:
        raise SystemExit("Provide an image path, e.g. python predict.py path/to/image.jpg")
    try:
        predict_image(args.image.resolve(), args.model if args.model.is_absolute() else ROOT_DIR / args.model)
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
