# Camera Trajectory Classifier (Few-Shot Learning with ResNet18)

This project implements a ResNet18-based image classification pipeline to infer camera motion types (e.g., zoom, orbit, turn) from static images. The classifier is trained in a few-shot learning setting and supports inference on individual images or batch image folders.

## Project Structure

```
├── train.py                         # Train ResNet18 classifier with dropout and AdamW
├── batch_predict.py                # Predict camera motion from a folder of test images
├── requirements.txt                # Python dependencies
├── camera_trajectory_resnet18.pth  # (Generated) Trained model weights
├── auto_coco_dataset/              # Dataset folder: ImageFolder format
│   ├── train/
│   │   ├── zoom_in/
│   │   ├── turn_left/
│   │   └── ...
│   └── val/
│       ├── zoom_in/
│       ├── turn_left/
│       └── ...
```

> Each class folder should contain at least 5–10 representative images.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Environment Setup

To replicate the environment:

```bash
conda create -n camera-env python=3.9
conda activate camera-env
pip install -r requirements.txt
```

## Training

1. **Train the model** (optional if you already have `camera_trajectory_resnet18.pth`):

    ```bash
    python train.py
    ```

2. **Run batch prediction on test images**:

    ```bash
    python batch_predict.py test_images/
    ```

## Inference (`batch_predict.py`)

### Usage

```bash
python batch_predict.py <image_folder>
```

### Output

- `predictions.csv`: contains filename, predicted class, and confidence
- `labeled_images/`: input images with predicted class labels drawn on top
