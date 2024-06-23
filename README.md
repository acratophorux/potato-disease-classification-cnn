# Potato Disease Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify diseases in potato plants based on leaf images. The model can identify healthy plants and detect two common potato diseases: Early Blight and Late Blight.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Potato diseases can significantly impact crop yield and quality. Early detection and classification of these diseases are crucial for effective management and prevention of crop losses. This project uses deep learning techniques, specifically Convolutional Neural Networks, to automate the process of identifying potato plant diseases from images of their leaves.

## Features

- Image preprocessing and augmentation
- Custom CNN architecture for potato disease classification
- Training script with configurable hyperparameters
- Evaluation metrics including accuracy, precision, recall, and F1-score
- Prediction script for classifying new images

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/potato-disease-classification.git
   cd potato-disease-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To train the model:
   ```
   python train.py --data_dir /path/to/dataset --epochs 50 --batch_size 32
   ```

2. To evaluate the model:
   ```
   python evaluate.py --model_path /path/to/saved/model --test_dir /path/to/test/data
   ```

3. To predict on a single image:
   ```
   python predict.py --image_path /path/to/image.jpg --model_path /path/to/saved/model
   ```

## Model Architecture

The CNN architecture used in this project consists of:
- Input layer
- 3 Convolutional layers with ReLU activation
- Max Pooling layers
- Dropout for regularization
- Flatten layer
- Dense layers
- Output layer with softmax activation

## Dataset

The dataset used for this project is the "PlantVillage" dataset, focusing on potato plant images. It includes images of healthy potato leaves and leaves affected by Early Blight and Late Blight.

## Training

The model is trained using the Adam optimizer and categorical cross-entropy loss. Data augmentation techniques such as rotation, flipping, and zooming are applied to increase the diversity of the training set and improve model generalization.

## Evaluation

The model's performance is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

A confusion matrix is also generated to visualize the model's performance across different classes.

## Results

(Include your model's performance results here after training and evaluation)

## Future Improvements

- Implement transfer learning using pre-trained models like ResNet or VGG
- Expand the dataset to include more diverse images and additional potato diseases
- Develop a web or mobile application for easy use in the field
- Investigate the use of explainable AI techniques to provide insights into model decisions

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
