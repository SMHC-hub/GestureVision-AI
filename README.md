# ✋ GestureVision-AI
### Real-Time 36-Class Alphanumeric Hand Gesture Recognition Using MediaPipe Landmarks and Deep CNNs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Enabled-brightgreen.svg)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org)

An end-to-end computer vision pipeline and real-time hand gesture recognition system built with Convolutional Neural Networks (CNN), MediaPipe 3D hand landmark extraction, and OpenCV.

## Overview

This project provides end-to-end workflows for classifying hand gestures from image streams and real-time webcam feeds. The model is trained on a custom hand gesture dataset categorized into 36 distinct alphanumeric classes (`0-9` and `A-Z`).

## Project Structure

```
Gesture-Recognition-Live-Using-Custom-Dataset/
├── gesture_model.keras          # Trained gesture recognition baseline model
├── gesture_model_final.keras    # Optimized gesture recognition model
├── Gesture.ipynb                # Complete notebook for dataset processing, training, & evaluation
├── Gesture Image Data/          # Custom training & validation gesture images (organized 0-9, A-Z)
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
└── .gitignore                   # Version control ignore specifications
```

## Requirements

- Python 3.8+
- TensorFlow / Keras 2.x
- MediaPipe
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Jupyter Notebook / JupyterLab

## Installation

```bash
# Clone the repository
git clone https://github.com/SMHC-hub/Gesture-Recognition-Live-Using-Custom-Dataset.git
cd Gesture-Recognition-Live-Using-Custom-Dataset

# Install dependencies
pip install tensorflow keras numpy opencv-python mediapipe matplotlib jupyter
```

## Usage

### 1. Training & Evaluation
Open the interactive Jupyter notebook:
```bash
jupyter notebook Gesture.ipynb
```
Follow the notebook cells to:
- Load, augment, and preprocess the custom gesture dataset.
- Construct the Convolutional Neural Network architecture.
- Train with early stopping and learning rate scheduling.
- Evaluate validation accuracy, loss curves, and confusion matrices.

### 2. Live Hand Gesture Inference
The notebook includes real-time hand landmark extraction via **MediaPipe** paired with the trained `.keras` model for live webcam inference and bounding-box gesture prediction.

## Model Details

- **Architecture**: Deep Convolutional Neural Network (Conv2D, MaxPooling, Batch Normalization, Dropout, Dense)
- **Input**: Preprocessed $64 \times 64 \times 3$ RGB gesture images
- **Output**: Softmax classification across 36 alphanumeric classes (`0-9`, `A-Z`)
- **Optimization**: Adam optimizer with categorical cross-entropy loss

## License

This project is open-source under the [MIT License](LICENSE).
