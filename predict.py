import os
import sys
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("models", "imageclassifier.keras")
CLASS_NAMES = ["angry", "happy", "sad"]
IMAGE_SIZE = (256, 256)

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = tf.image.resize(img_rgb, IMAGE_SIZE)
    img_tensor = np.expand_dims(img_resized / 255.0, axis=0)

    # Inference
    probabilities = model.predict(img_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx]

    print("\n" + "=" * 35)
    print(f"File:       {image_path}")
    print(f"Prediction: {predicted_class.upper()} ({confidence * 100:.2f}%)")
    print("-" * 35)
    for name, prob in zip(CLASS_NAMES, probabilities):
        print(f" - {name:<7s}: {prob * 100:6.2f}%")
    print("=" * 35 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        # Default fallback test on an existing image in your dataset
        sample_path = os.path.join("images", "happy", os.listdir(os.path.join("images", "happy"))[0])
        predict_image(sample_path)