# Deep Learning & Its Applications

My coursework for the **5C16 Deep Learning & Its Applications** module at Trinity College Dublin. All models are written in PyTorch.

## Contents

| Folder | Topic | Summary |
|---|---|---|
| [Feedforward Fully Connected Neural Network [CIFAR-10]](Feedforward%20Fully%20Connected%20Neural%20Network%20%5BCIFAR-10%5D) | Feedforward networks | A fully connected classifier for CIFAR-10 (32×32 RGB images, 10 classes). It has three hidden layers in a funnel shape (1024 → 512 → 256) with ReLU activations and dropout, and about 3.8M parameters, under the 5M limit. |
| [Convolutional Neural Network [CIFAR-10]](Convolutional%20Neural%20Network%20%5BCIFAR-10%5D) | Convolutional neural networks | A CNN classifier for CIFAR-10 with three conv → ReLU → max-pool stages (64 → 128 → 256 channels), followed by a dense classifier head. It uses dropout in both the convolutional and dense layers. |
| [Fast Super-Resolution CNN [x4 Upscaling]](Fast%20Super-Resolution%20CNN%20%5Bx4%20Upscaling%5D) | Mini project: super-resolution | A real-time ×4 super-resolution model that upscales 32×32 images to 128×128. A small CNN learns the detail to add on top of a bicubic upscale, using 3×3 convolutions at low resolution and two transposed convolutions. It beats bicubic by +1.30 dB PSNR with 245k parameters, at 0.47 ms per frame. The folder's README documents the 11 development runs. |

Each folder contains a notebook (`.ipynb`) with the training and evaluation code, and `models.py` with the model definition.

## Running

The CIFAR-10 notebooks were written for Google Colab with a GPU runtime. You need Python 3 with `torch`, `torchvision`, `numpy` and `matplotlib`.

The super-resolution project was trained locally on an NVIDIA RTX 5070. Its folder has a `requirements.txt` and setup steps.

Datasets and large data files are not included in this repo. The notebooks download them when run.
