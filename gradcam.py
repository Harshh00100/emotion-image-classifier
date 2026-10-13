"""Create a Grad-CAM heatmap for the custom CNN classifier.

Example:
    python gradcam.py path/to/face.jpg
    python gradcam.py path/to/face.jpg --output reports/face_gradcam.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (256, 256)
DEFAULT_MODEL = ROOT_DIR / "models" / "imageclassifier.keras"


def main():
    parser = argparse.ArgumentParser(description="Create a Grad-CAM visualization for the custom CNN.")
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to a custom CNN .keras model.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    args = parser.parse_args()

    image_path = args.image.resolve()
    model_path = args.model if args.model.is_absolute() else ROOT_DIR / args.model
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}. Train image_classifier.py first.")

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            pixels = np.asarray(image, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise SystemExit(f"Could not open image {image_path}: {exc}") from exc

    model = tf.keras.models.load_model(model_path)
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise SystemExit(
            "This model does not expose a top-level Conv2D layer. "
            "Grad-CAM currently supports models trained with image_classifier.py."
        )
    last_conv = conv_layers[-1]
    grad_model = tf.keras.Model(model.inputs, [last_conv.output, model.output])

    image_batch = np.expand_dims(pixels, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]
    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise SystemExit("Grad-CAM gradients could not be computed for this model.")
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap)).numpy()

    images_dir = ROOT_DIR / "images"
    class_names = sorted(path.name for path in images_dir.iterdir() if path.is_dir()) if images_dir.is_dir() else []
    predicted_index = int(class_index.numpy())
    label = class_names[predicted_index] if predicted_index < len(class_names) else str(predicted_index)
    confidence = float(predictions.numpy()[0, predicted_index])

    output_path = args.output or image_path.with_name(f"{image_path.stem}_gradcam.png")
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = tf.image.resize(heatmap[..., np.newaxis], IMAGE_SIZE).numpy().squeeze()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pixels.astype(np.uint8))
    axes[0].set_title("Input image")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Heatmap: {label}")
    axes[2].imshow(pixels.astype(np.uint8))
    axes[2].imshow(overlay, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    axes[2].set_title(f"Overlay: {label} ({confidence:.1%})")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction: {label} ({confidence:.2%})")
    print(f"Grad-CAM saved to: {output_path}")


if __name__ == "__main__":
    main()
