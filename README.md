# Unsupervised Deep Learning Project Repo
Project Repository for the 2024/2025 Unsupervised Deep Learning course at the University of Groningen

## Project overview
TODO: work out. We want to basically train a diffusion model, but see if we can improve it by taking some inspiration from the GAN-based literature.

## Repository overview
The file structure of this repo is as follows:
- `udl_module/`: a Python package containing most of the code
  - `nn/`: we can play around with different neural network architectures. All PyTorch `nn.Module` objects should go here.
  - `jiri_diffusion/`: Jiri's experiments for training a baseline diffusion model.
  - `vincent_diffusion/`: Vincent's experiments for training a baseline diffusion model.
  - `jiri_gan/`: Jiri's experiments to introduce GAN-based loss into the diffusion model.
  - `vincent_wgan/` Vincent's experiments to introduce Wasserstein-based GAN loss into the diffusion model.
- `notebooks/`: The Jupyter Notebooks we write for training. These should import the `udl_module` code.

## Some ideas to already write down
We can see if a discriminator can discriminate between the diffusion process and the noising process. This means we should give any pair $`\mathbf{z}_t`$ and $`\mathbf{z}_{t+1}`$ as input to the discriminator. We can see if we can create a combined loss function, that is, for example, the average of the diffusion and the GAN loss, and otherwise switch back and forward or something. We will see how it goes 🤓
