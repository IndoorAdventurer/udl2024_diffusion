from ..nn import DiffusionNoiser

import torch
from torch import nn

import lightning as L


class DiffusionModel(L.LightningModule):

    def __init__(   self,
                    model: nn.Module,
                    optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
                    optimizer_args: dict = dict(),
                    steps: int = 1000,
                    beta_start: float = 1e-4,
                    beta_end: float = 0.02
                 ):
        """
        Args:
            model: the underlying denoising architecture (typically a U-Net). It
                should take a batch of noisy images and integer time values as
                input to the `forward` method.
            optimizer_cls: the class type of the optimizer
            optimizer_args: the args (other than params) to give the optimizer
                at init.
            steps: number of noising steps to use
            beta_start: the initial value for beta (linear interpolation)
            beta_end: the final value of beta (linear interpolation)
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.noiser = DiffusionNoiser(steps, beta_start, beta_end)
    
    def forward(self, x, t):
        return self.model(x, t)
    
    def training_step(self, batch, batch_idx):
        noise = torch.randn_like(batch)
        ts = torch.randint(0, len(self.noiser.betas), (len(batch),), device=batch.device)
        noised_img = self.noiser.closed_form_noise(batch, noise, ts)

        noise_hat = self.model(noised_img, ts)

        loss = nn.functional.mse_loss(noise_hat, noise)
        self.log("train/mse", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), *self.optimizer_args)
    
    @torch.inference_mode()
    def sample_img(self, shape: tuple[int]):
        """Samples from a trained diffusion model.
        
        Args:
            shape: the shape of the element or batch [(B x) C x W x H]
        """
        self.eval()
        noisy_img = torch.randn(shape, device=self.device)
        steps = len(self.noiser.betas)
        for idx in reversed(range(steps)):
            t = torch.tensor([idx], device=self.device)
            predicted_noise = self(noisy_img, t)
            new_noise = torch.randn_like(noisy_img)
            noisy_img = self.noiser.denoising_step(noisy_img, predicted_noise, t, new_noise)
        return noisy_img