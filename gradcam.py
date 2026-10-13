"""Generate a Grad-CAM heatmap for a trained emotion classifier.

Examples:
    python gradcam.py path/to/face.jpg
    python gradcam.py path/to/face.jpg --model models/mobilenetv2_emotion.keras
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (256, 256)
DEFAULT_MODEL = ROOT_DIR / "models" / "imageclassifier.keras"


def load_image(path: Path):
    with Image.open(path) as source:
        image = source.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    return np.asarray(image, dtype=np.float32)


def find_last_conv_layer(model):
    # Custom CNN exposes Conv2D layers directly. MobileNetV2 nests its feature
    # extractor, so search nested models from the inside out as a fallback.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return model, layer.name
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                nested_model, layer_name = find_last_conv_layer(layer)
                return nested_model, layer_name
            except ValueError:
                pass
    raise ValueError("No Conv2D layer found in this model; Grad-CAM is unsupported.")


def make_gradcam_heatmap(model, image_batch, class_index=None):
    target_model, conv_layer_name = find_last_conv_layer(model)
    grad_model = tf.keras.Model(
        inputs=target_model.inputs,
        outputs=[target_model.get_layer(conv_layer_name).output, target_model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]
    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise ValueError("Could not compute gradients for the selected convolution layer.")
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))
    return heatmap.numpy(), int(class_index.numpy()), predictions.numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="Create a Grad-CAM visualization.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Trained .keras model.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    args = parser.parse_args()
    image_path = args.image.resolve()
    model_path = args.model if args.model.is_absolute() else ROOT_DIR / args.model
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}. Train a model first.")

    pixels = load_image(image_path)
    model = tf.keras.models.load_model(model_path)
    heatmap, class_index, probabilities = make_gradcam_heatmap(model, np.expand_dims(pixels, axis=0))
    class_names = sorted(p.name for p in (ROOT_DIR / "images").iterdir() if p.is_dir())
    label = class_names[class_index] if class_index < len(class_names) else str(class_index)
    output_path = args.output or image_path.with_name(f"{image_path.stem}_gradcam.png")
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pixels.astype(np.uint8))
    axes[0].set_title("Input image")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM: {label} ({probabilities[class_index] * 100:.1f}%)")
    axes[2].imshow(pixels.astype(np.uint8))
    axes[2].imshow(
        tf.image.resize(heatmap[..., np.newaxis], IMAGE_SIZE).numpy().squeeze(),
        cmap="jet", alpha=0.4, vmin=0, vmax=1,
    )
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM saved to: {output_path}")


if __name__ == "__main__":
    main()
