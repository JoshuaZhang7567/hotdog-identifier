# Hotdog Detection with YOLO

This repository contains code and data for training and running a YOLO-based hotdog detector using the [Ultralytics YOLO](https://docs.ultralytics.com/) library. This project came from an exercise provided by aUToronto: https://docs.google.com/document/d/1LOAFpgdQ3vUs3Wie1m9g8ndhfikjZxD7HOQEDznBSJQ

## Folder Structure

- `training/` — Training scripts, dataset, and labels.
- `prediction/` — Inference script and saved model weights.

## Setup

1. **Install dependencies:**

   ```sh
   pip install ultralytics opencv-python
   ```

2. **Train the model:**

Run the training script in `training/hotdog_detection_training.py`

3. **Transfer weights:**

After training, move the exported weights (e.g., hotdog_model.pt) to the prediction/ folder and rename to weights.pt if necessary.

## Running Predictions
Use the webcam prediction script:
`hotdog_prediction.py`

This will open your webcam and display live hotdog detections.

## Notes
- Make sure the weights file (weights.pt) is present in the prediction/ folder before running predictions.
- The dataset and labels are organized in YOLO format under training/data/.

For more details on Ultralytics YOLO, see the official documentation.
