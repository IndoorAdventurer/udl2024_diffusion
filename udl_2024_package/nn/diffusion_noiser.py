# Vincent Tonkes 2025

import torch
from torch import nn


class DiffusionNoiser(nn.Module):
    """A class to define the noising process. Will have methods to add noise
    in the closed and open form to images, but also inverse operations, such as
    removing (predicted) noise from an image, or sampling a new image.
    
    My convention for indexing is as follows:
    - t=0 is the original image.
    - t=steps(1000) should be pure gaussian noise
    - take image of step t and alphas/betas of step t to go to t+1
    - this means alphas and betas go from t=0 to t=steps(1000)-1

    Turned it into a nn.Module such that I don't need to worry about putting
    tensors on the correct device, etc.
    """

    def __init__(   self,
                    steps: int = 1000,
                    beta_start: float = 1e-4,
                    beta_end: float = 0.02
                 ):
        """
        Args:
            steps: number of noising steps to use
            beta_start: the initial value for beta (linear interpolation)
            beta_end: the final value of beta (linear interpolation)
        """
        super().__init__()

        betas = torch.linspace(beta_start, beta_end, steps)
        alpha_bars = torch.cumprod(1 - betas, 0)

        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
    
    def forward(self, img, noise, t):
        """Calls DiffusionNoiser.closed_form_noise()."""
        return self.closed_form_noise(img, noise, t)

    def closed_form_noise(self, img, noise, t):
        """Adds noise to an image using the DDPM closed form formula.

        Args:
            img: The image(s) to add noise to [(B x) C x H x W].
            noise: the gaussian noise to add N(0,1) [(B x) C x H x W].
            t: integer time step (single integer or shape [B,])
        """
        alpha_bar = self.reshape(self.alpha_bars[t], img)
        return torch.sqrt(alpha_bar) * img + torch.sqrt(1 - alpha_bar) * noise

    def noise_from_closed_form_noise(self, img, noised_img, t):
        """Inverse of closed_form_noise: returns the noise, given the original
        imaged and a noised version.
        
        Args:
            img: the original image.
            noised_img: a noised version of the image.
            t: the timestep used to get from img to noised_img.
        """
        alpha_bar = self.reshape(self.alpha_bars[t], img)
        return (noised_img - torch.sqrt(alpha_bar) * img) / torch.sqrt(1 - alpha_bar)
    
    def img_from_closed_form_noise(self, noised_img, noise, t):
        """Inverse of closed_form_noise: returns the original image, given the
        noise and a noised version of the image.
        
        Args:
            noised_img: a noised version of the image.
            noise: the noise that was added to get noised_img.
            t: the timestep used to get from imgage to noised_img using noise.
        """
        alpha_bar = self.reshape(self.alpha_bars[t], noised_img)
        return (noised_img - torch.sqrt(1 - alpha_bar) * noise ) / torch.sqrt(alpha_bar)
    
    def forward_noise_step(self, img_prev, noise, t):
        """The forward noising process in a step-wise manner: computes a
        slightly noisier image from img_prev.
        
        Args:
            img_prev: image corresponding to step t.
            noise: the noise to add to get to step t+1.
            t: current time step."""
        beta = self.reshape(self.betas[t], img_prev)
        return torch.sqrt(1 - beta) * img_prev + torch.sqrt(beta) * noise
    
    def denoising_step(self, img_t, noise_t, t, new_noise):
        """The inverse process of the forward_noise_step: predicts image at
        step t from image at step t+1. Using formulas 6 and 7 from the original
        paper https://arxiv.org/pdf/2006.11239
        
        Args:
            img_next: image from step t+1.
            noise_next: the (predicted) noise in img_next.
            t: current time step.
            new_noise: new pure gausssian noise N(0, 1) to add.
        """
        prev_t = torch.clamp(t - 1, 0)
        
        beta = self.reshape(self.betas[t], img_t)
        alpha_bar = self.reshape(self.alpha_bars[t], img_t)
        alpha_bar_prev = self.reshape(self.alpha_bars[prev_t], img_t)

        img_0 = self.img_from_closed_form_noise(img_t, noise_t, t)
        img_0 = img_0.clamp(-10, 10) # This does not seem to be needed.
        
        img_0_scalar = torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)
        img_t_scalar = torch.sqrt(1 - beta) * (1 - alpha_bar_prev) / (1 - alpha_bar)
        mu_t = img_0_scalar * img_0 + img_t_scalar * img_t

        variance_t = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        
        return mu_t + torch.sqrt(variance_t) * new_noise
    
    def reshape(self, scalars: torch.Tensor, tensor: torch.Tensor):
        """Adds dimensions to scalars so they broadcast properly."""
        b = len(scalars)
        extra_dims = len(tensor.shape) - 1
        return scalars.view(b, *([1] * extra_dims))