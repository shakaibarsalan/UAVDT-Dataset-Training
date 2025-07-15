# UAVDT-Dataset-Training

A PyTorch-based pipeline for training and evaluating object detection models on the UAVDT dataset using Faster R-CNN. This project provides custom dataset handling, training, evaluation, and advanced visualization utilities for UAV-based vehicle detection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
- [Customization](#customization)
- [Results](#results)
- [License](#license)

## Overview

This repository enables training and evaluation of a Faster R-CNN model on the UAVDT dataset, a challenging benchmark for object detection in aerial images. The code is designed for easy use in both Kaggle and Colab environments, with modular components for dataset parsing, model training, and result visualization.

## Features

- Custom PyTorch Dataset for UAVDT-style annotation parsing
- Faster R-CNN (ResNet-50 FPN) with configurable number of classes
- Flexible DataLoader with custom collate and transform functions
- Training loop with checkpointing and learning rate scheduling
- Evaluation metrics (detections per image, thresholding)
- Visualization tools for predictions, ground truth, and side-by-side comparisons
- Support for batch size, number of images, and device selection

## Dataset

- **UAVDT**: Unmanned Aerial Vehicle Detection dataset
- Download via Kaggle: `shakaibkaggle/uavdt-dataset`
- Directory structure:
  ```
  /uavdt-dataset/
    ├── train/
    │   ├── img/
    │   └── ann/
    └── test/
        ├── img/
        └── ann/
  ```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/uavdt-dataset-training.git
   cd uavdt-dataset-training
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision numpy matplotlib tqdm pillow kagglehub
   ```

3. **(Optional) Download the dataset via Kaggle:**
   ```python
   import kagglehub
   kagglehub.dataset_download('shakaibkaggle/uavdt-dataset')
   ```

## Usage

### Data Preparation

- Ensure the UAVDT dataset is downloaded and extracted in the correct folder structure.
- Update `dataset_path` in the script to point to your dataset location.

### Training

```python
from your_module import train_model, get_model, UAVDTDataset, get_transforms

# Prepare dataset and dataloader
train_dataset = UAVDTDataset(root_dir='/path/to/uavdt-dataset', split='train', transforms=get_transforms(train=True))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)

# Initialize model
num_classes = len(train_dataset.class_to_idx)
model = get_model(num_classes)

# Train
trained_model = train_model(model, train_loader, optimizer, lr_scheduler, num_epochs=10)
```

### Evaluation

```python
from your_module import evaluate_model

# Evaluate on test set
evaluate_model(trained_model, test_loader, device)
```

### Visualization

- **Visualize predictions:**
  ```python
  visualize_predictions(trained_model, test_dataset, num_samples=3)
  ```

- **Advanced visualization:**
  ```python
  visualize_object_detection_results(
      model=trained_model,
      test_dataset=test_dataset,
      num_samples=6,
      confidence_threshold=0.5
  )
  ```

- **Side-by-side comparison:**
  ```python
  visualize_detection_comparison(
      model=trained_model,
      test_dataset=test_dataset,
      num_samples=4,
      confidence_threshold=0.5
  )
  ```

## Customization

- Change number of images: Use `max_images` parameter in `UAVDTDataset`.
- Adjust batch size: Modify `batch_size` in DataLoader.
- Set number of epochs: Change `num_epochs` in `train_model`.
- Modify transforms: Edit `get_transforms()` for data augmentation.
- Change model: Swap out `get_model()` for a different architecture if desired.

## Results

- Model checkpoints are saved every 5 epochs and at the end of training.
- Evaluation prints average detections per image and other metrics.
- Visualizations provide qualitative insights into model performance, including detection overlays and comparison plots.


## Acknowledgements

- UAVDT Dataset: UAVDT Benchmark
- PyTorch, torchvision, and Kaggle
