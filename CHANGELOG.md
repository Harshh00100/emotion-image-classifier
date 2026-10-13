# Changelog

Notable changes to the Emotion Image Classifier are recorded here.

## Unreleased — issue-fix pull request

### Fixed
- Replaced the removed `imghdr` dependency with Pillow-based image verification.
- Changed invalid-image handling to report-only by default. Training stops when invalid supported image files are found; deletion requires the explicit `--delete-invalid` flag.
- Changed dataset splitting to operate on individual images rather than batches.
- Corrected the MobileNetV2 classifier head to use three-class softmax probabilities with sparse categorical cross-entropy.
- Saved MobileNetV2 to `models/mobilenetv2_emotion.keras`, separately from the custom CNN at `models/imageclassifier.keras`.
- Aligned prediction input handling with both models' embedded preprocessing and added the `--model` option.

### Added
- `gradcam.py` for Grad-CAM heatmaps and image overlays for the custom CNN.
- Documentation for training, prediction, validation behavior, Grad-CAM usage, and current dataset limitations.

### Validation status
The source changes were committed to the `fix/open-issues-10-11` branch. Full TensorFlow training and inference should be run locally before treating the changes as runtime-verified.
