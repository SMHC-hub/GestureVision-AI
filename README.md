# Gesture Recognition

A deep learning project for recognizing hand gestures using convolutional neural networks (CNN).

## Overview

This project uses a Keras/TensorFlow model to classify hand gestures from images. The model is trained on a dataset of gesture images organized by class (0-9, A-Z).

## Project Structure

```
Gesture Recognition/
├── gesture_model.keras          # Trained gesture recognition model
├── gesture_model_final.keras    # Final optimized model
├── Gesture.ipynb                # Jupyter notebook with model training and evaluation
├── Gesture Image Data/           # Training dataset
│   ├── 0-9/                     # Numeric gesture classes
│   └── A-Z/                     # Alphabetic gesture classes
├── README.md                    # This file
└── .gitignore                   # Git ignore configuration
```

## Requirements

- Python 3.8+
- TensorFlow/Keras
- NumPy
- OpenCV (cv2)
- Matplotlib
- Jupyter

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Gesture Recognition"

# Install dependencies
pip install tensorflow keras numpy opencv-python matplotlib jupyter
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook Gesture.ipynb
```

2. Run the cells to:
   - Load and preprocess the gesture image dataset
   - Train the gesture recognition model
   - Evaluate model performance
   - Make predictions on new images

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Gesture images (preprocessed)
- **Output**: Classification into 36 classes (0-9, A-Z)
- **Framework**: TensorFlow/Keras

## Dataset

The dataset consists of hand gesture images organized into 36 classes:
- Digits: 0-9
- Letters: A-Z

Each gesture class should have multiple sample images for training.

## Results

The trained models (`gesture_model.keras` and `gesture_model_final.keras`) can be used for inference on new gesture images.

## License

[Add your license here]

## Author

[Your name/username]
