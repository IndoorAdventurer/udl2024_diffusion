# Unsupervised Deep Learning Project Repo

Project Repository for the 2024/2025 Unsupervised Deep Learning course at the University of Groningen

## Project overview
For this university AI project, we implemented `DDPM` and `(W)GAN`, and looked at two different ways to combine them. Firstly, we investigated transfer learning properties when fine-tuning a `DDPM` `U-Net` as `GAN` generator. Next, we explored if a discriminator model could distinguish between the forward and backward diffusion process, and if incorporating this in the diffusion loss would still make the model converge -- and if it would lead to any qualitative or quantitative differences.

## Repository overview
The file structure of this repo is as follows:
- `experiment_1/`: this folder contains `Python` notebooks related to the first experiment, related to transfer learning.
- `experiment_2`: this folder contains our implementation for the second experiment: combining `DDPM` and `GAN` loss.
- `udl_module/`: a Python package containing much of the code used in experiment 1. It has subfolders:
	- `nn/`: Our `PyTorch` `nn.Module` classes, such as a `U-Net` implementation.
	- `diffusion/`: Backbone of the `DDPM` implementation.
	- `wgan/`: Backbone of the Wasserstein GAN implementation.
	- `datasets/`: Everything related to loading training/testing data for our models.