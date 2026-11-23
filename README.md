# Deep Learning & Its Applications

My coursework for **4C16 Deep Learning & Its Applications** at Trinity College Dublin. All models are written in PyTorch.

## Contents

| Folder | Topic | Summary |
|---|---|---|
| [Feedforward Fully Connected Neural Network [CIFAR-10]](Feedforward%20Fully%20Connected%20Neural%20Network%20%5BCIFAR-10%5D) | Feedforward networks | A fully connected classifier for CIFAR-10 (32×32 RGB images, 10 classes). It has three hidden layers in a funnel shape (1024 → 512 → 256) with ReLU activations and dropout, and about 3.8M parameters, under the 5M limit. |
| [Convolutional Neural Network [CIFAR-10]](Convolutional%20Neural%20Network%20%5BCIFAR-10%5D) | Convolutional neural networks | A CNN classifier for CIFAR-10 with three conv → ReLU → max-pool stages (64 → 128 → 256 channels), followed by a dense classifier head. It uses dropout in both the convolutional and dense layers. |
| Super-Resolution Mini Project *(coming soon)* | Mini project: super-resolution | A real-time ×4 image super-resolution model that upscales 32×32 images to 128×128, under 5M parameters. |

Each folder contains a notebook (`.ipynb`) with the training and evaluation code, and `models.py` with the model definition.

## Running

The notebooks were written for Google Colab with a GPU runtime. You need Python 3 with `torch`, `torchvision`, `numpy` and `matplotlib`.

Datasets and large data files are not included in this repo. The notebooks download them when run.
