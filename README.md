# Deep Convolutional Generative Adversarial Network (DCGAN) for Image Generation

## Overview
This repository contains an implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** for generating high-quality images. The model is trained using deep learning techniques to generate realistic images by learning the underlying data distribution.

## Dataset
The dataset consists of a collection of images used to train the DCGAN. These images undergo preprocessing to enhance training efficiency.

## Preprocessing
- Images are loaded using `PIL` and `torchvision.transforms`.
- Resized to a fixed dimension suitable for the model.
- Normalized to a range of [-1,1] for better convergence.
- Converted into PyTorch tensors for GPU acceleration.

## Model Architecture
DCGAN consists of:
- **Generator**: A deep convolutional neural network that generates realistic images from random noise.
- **Discriminator**: A convolutional classifier that distinguishes between real and generated images.
- **Loss Function**: Uses Binary Cross-Entropy (BCE) loss to optimize both networks.

The architecture follows best practices for training GANs using deep convolutional layers, batch normalization, and LeakyReLU activations.

## Training
- The model is trained using **adversarial training**, where the Generator and Discriminator compete to improve generation quality.
- Uses the **Adam optimizer** with tuned hyperparameters.
- Trained for multiple epochs with loss values monitored for both networks.
- Includes visualizations to track the progress of generated images over time.

## Evaluation
- Generated images are visually inspected for realism.
- Loss curves are plotted to evaluate training stability.
- The quality of generated images improves progressively with training.

## Results
- The model successfully generates high-quality images.
- Loss curves indicate stable adversarial training dynamics.
- The Generator learns to produce more detailed and realistic images over time.
