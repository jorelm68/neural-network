# Simple Neural Network for MNIST Classification

This repository contains code for a basic neural network designed to classify handwritten digits from the MNIST dataset. The code implements standard neural network functionality, including forward propagation, backpropagation, and adjustable activation and loss functions.

## Features

- **Customizable Network Architecture**: Configured with a hidden layer of 128 neurons and an output layer for multi-class classification.
- **Activation Functions**: Supports ReLU and Sigmoid activations for hidden layers.
- **Loss Functions**: Allows for both Cross-Entropy and Mean Squared Error losses to suit various tasks and preferences.
- **He Initialization**: Weight initialization method to improve convergence.
- **Training and Testing with MNIST**: Designed to train on MNIST’s handwritten digit dataset, with helper functions for data loading and processing.

## Code Overview

The code is organized with a clear structure to facilitate understanding and experimentation. Key components include:

### 1. Helper Functions
  - `reLU` and `sigmoid`: For hidden layer activations.
  - `crossEntropy` and `meanSquaredError`: Loss functions for output layer.
  - **MNIST Data Processing**: Code includes methods to load, normalize, and preprocess MNIST data for training and testing.

### 2. Forward and Backward Propagation
  - **Forward Pass**: Computes activations for each layer using the selected activation function.
  - **Backpropagation**: Computes gradients for each layer, adjusting weights and biases to minimize the loss.

### 3. Training Loop
  - **Initialization**: Sets up weights with He initialization.
  - **Batch Processing**: Divides data into batches, calculates loss and accuracy, and adjusts weights.
  - **Epoch Management**: Tracks loss and accuracy across epochs to monitor progress.

### 4. Evaluation and Testing
  - **Accuracy Measurement**: Calculates classification accuracy on the test set.
  - **Loss Monitoring**: Monitors loss throughout training to ensure convergence.

## Getting Started

### Prerequisites
- Python 3.x
- NumPy
- MNIST Dataset (downloaded via helper functions)

### Installation

Clone this repository:

```bash
git clone https://github.com/jorelm68/neural-network.git
cd mnist-neural-network
