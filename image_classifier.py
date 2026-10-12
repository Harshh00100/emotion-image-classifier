import os
import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
DATA_DIR = "images"
MODELS_DIR = "models"
LOGS_DIR = "logs"
MODEL_PATH = os.path.join(MODELS_DIR, "imageclassifier.keras")
CONTACT_SHEET_PATH = "dataset_contact_sheet.jpg"
TRAINING_PLOT_PATH = "training_performance.png"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42
VALID_IMAGE_EXTENSIONS = ["jpeg", "jpg", "bmp", "png"]

# -----------------------------------------------------------------------------
# 1. Image Sanitization & Dataset Validation
# -----------------------------------------------------------------------------
def sanitize_image_directory(data_dir):
    """
    Scans the dataset directory, removes corrupt or non-decodable images,
    and ensures valid image extensions.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    removed_count = 0
    total_checked = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_checked += 1
            try:
                ext = imghdr.what(file_path)
                if ext not in VALID_IMAGE_EXTENSIONS:
                    print(f"Removing invalid extension: {file_path}")
                    os.remove(file_path)
                    removed_count += 1
                    continue

                # Verify actual decoding capability with OpenCV
                img = cv2.imread(file_path)
                if img is None or img.size == 0:
                    print(f"Removing unreadable image: {file_path}")
                    os.remove(file_path)
                    removed_count += 1
            except Exception as e:
                print(f"Error processing {file_path} ({e}). Removing.")
                if os.path.exists(file_path):
                    os.remove(file_path)
                removed_count += 1

    print(f"Sanitization complete: Checked {total_checked} files, pruned {removed_count} corrupt items.")


# -----------------------------------------------------------------------------
# 2. Contact Sheet Generation
# -----------------------------------------------------------------------------
def generate_contact_sheet(data_dir, output_path=CONTACT_SHEET_PATH, samples_per_class=4):
    """
    Creates a visual contact sheet sampling images from each emotion class.
    """
    classes = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
    num_classes = len(classes)
    if num_classes == 0:
        return

    plt.figure(figsize=(samples_per_class * 3, num_classes * 3))

    plot_idx = 1
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(tuple(VALID_IMAGE_EXTENSIONS))]
        selected_files = files[:samples_per_class]

        for file in selected_files:
            file_path = os.path.join(cls_dir, file)
            img = cv2.imread(file_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(num_classes, samples_per_class, plot_idx)
                plt.imshow(img_rgb)
                plt.title(cls)
                plt.axis("off")
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Contact sheet saved to: {os.path.abspath(output_path)}")


# -----------------------------------------------------------------------------
# 3. Data Loading & Partitioning Pipeline
# -----------------------------------------------------------------------------
def build_datasets():
    """
    Loads images dynamically, computes per-class distribution splits,
    and returns optimized tf.data.Dataset objects.
    """
    raw_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    class_names = raw_dataset.class_names
    num_classes = len(class_names)
    print(f"Class names (index order): {class_names}")

    # Normalize pixel channels: [0, 255] -> [0.0, 1.0]
    scaled_dataset = raw_dataset.map(lambda x, y: (x / 255.0, y))

    # Calculate partition boundaries
    total_batches = len(scaled_dataset)
    train_size = int(total_batches * 0.7)
    val_size = int(total_batches * 0.2)
    test_size = total_batches - train_size - val_size

    train_data = scaled_dataset.take(train_size)
    val_data = scaled_dataset.skip(train_size).take(val_size)
    test_data = scaled_dataset.skip(train_size + val_size).take(test_size)

    # Count class totals across splits
    def count_classes(ds):
        counts = {cls: 0 for cls in class_names}
        for _, labels in ds:
            labels_np = labels.numpy()
            indices = np.argmax(labels_np, axis=1)
            for idx in indices:
                counts[class_names[idx]] += 1
        return counts

    train_counts = count_classes(train_data)
    val_counts = count_classes(val_data)
    test_counts = count_classes(test_data)

    print(f"Total batches: {total_batches} | Train: {train_size} | Validation: {val_size} | Test: {test_size}")
    print(f"Train class counts:      {train_counts}")
    print(f"Validation class counts: {val_counts}")
    print(f"Test class counts:       {test_counts}")

    # Optimize pipeline caching & prefetching
    autotune = tf.data.AUTOTUNE
    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_data = val_data.cache().prefetch(buffer_size=autotune)
    test_data = test_data.cache().prefetch(buffer_size=autotune)

    return train_data, val_data, test_data, class_names


# -----------------------------------------------------------------------------
# 4. Model Architecture & Compilation
# -----------------------------------------------------------------------------
def build_model(num_classes):
    """
    Constructs the CNN model with data augmentation and multiclass head.
    """
    model = models.Sequential([
        # Data Augmentation Layers
        layers.RandomFlip("horizontal", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.RandomRotation(0.1),

        # Feature Extraction Backbone
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),

        # Dense Classifier Head
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    )
    return model


# -----------------------------------------------------------------------------
# 5. Training History Plotting
# -----------------------------------------------------------------------------
def plot_training_history(history, output_path=TRAINING_PLOT_PATH):
    """
    Plots and saves loss and accuracy learning curves across epochs.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="blue")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="orange")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="blue")
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training performance plot saved to: {os.path.abspath(output_path)}")


# -----------------------------------------------------------------------------
# 6. Evaluation, Metrics & Inference
# -----------------------------------------------------------------------------
def evaluate_and_test(model, test_data, class_names):
    """
    Calculates overall and per-class precision, recall, and accuracy,
    then executes single-image test inference.
    """
    print("\nTest Results")
    print("-" * 35)

    y_true = []
    y_pred = []

    for images, labels in test_data:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) > 0:
        overall_accuracy = np.mean(y_true == y_pred)
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print("\nPer-Class Metrics:")

        for idx, cls in enumerate(class_names):
            tp = np.sum((y_pred == idx) & (y_true == idx))
            fp = np.sum((y_pred == idx) & (y_true != idx))
            fn = np.sum((y_pred != idx) & (y_true == idx))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            print(f" - {cls:<7s} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Single-image inference test
    sample_img_path = None
    for cls in class_names:
        folder = os.path.join(DATA_DIR, cls)
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(tuple(VALID_IMAGE_EXTENSIONS))]
            if files:
                sample_img_path = os.path.join(folder, files[0])
                break

    if sample_img_path:
        img_raw = cv2.imread(sample_img_path)
        if img_raw is not None:
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_resized = tf.image.resize(img_rgb, IMAGE_SIZE)
            img_input = np.expand_dims(img_resized / 255.0, axis=0)

            predictions = model.predict(img_input, verbose=0)[0]
            predicted_idx = int(np.argmax(predictions))
            predicted_class = class_names[predicted_idx]
            confidence = predictions[predicted_idx]

            print("\nSample Test Inference:")
            print(f"Image:      {sample_img_path}")
            print(f"Prediction: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            print("\nProbability Distribution:")
            for name, prob in zip(class_names, predictions):
                print(f" - {name:<7s}: {prob * 100:.2f}%")


# -----------------------------------------------------------------------------
# 7. Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    # 1. Clean data and build contact sheet
    sanitize_image_directory(DATA_DIR)
    generate_contact_sheet(DATA_DIR)

    # 2. Partition dataset
    train_data, val_data, test_data, class_names = build_datasets()
    num_classes = len(class_names)

    # 3. Build & summarize model
    model = build_model(num_classes)
    model.summary()

    # 4. Setup TensorBoard and callbacks
    os.makedirs(LOGS_DIR, exist_ok=True)
    tb_callback = callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)

    # 5. Train model
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        callbacks=[tb_callback]
    )

    # 6. Save plots and run full multiclass evaluation
    plot_training_history(history)
    evaluate_and_test(model, test_data, class_names)

    # 7. Persist model artifact
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel successfully saved to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()