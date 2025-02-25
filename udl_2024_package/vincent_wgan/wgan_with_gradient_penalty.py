import torch
from torch import nn
import lightning as L
from .wgan_funcs import WGANWithGradientPenaltyFuncs


class WGANWithGradientPenalty(L.LightningModule):
    """Wasserstein Gan that uses Gradient Penalty instead of weight clipping."""

    def __init__(
            self,
            generator: nn.Module,
            critic: nn.Module,
            generator_func,
            gp_weight: float = 10,
            critic_iterations: int = 5,
            optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
            gen_optimizer_args: dict = {"lr": 1e-4, "betas": (0.5, 0.99)},
            cri_optimizer_args: dict = {"lr": 1e-4, "betas": (0.5, 0.99)},
    ):
        """
        Args:
            generator: the generator model for the WGAN
            critic: the critic model for the WGAN. Outputs single scalar value
            generator_func: function `generator_func(generator, batch)` that
                takes the generator and a batch (to get right shape) as input
                and outputs generated images.
            gp_weight: gradient penalty weight. See `WGANWithGradientPenaltyFuncs`
            critic_iterations: See `WGANWithGradientPenaltyFuncs`
            optimizer_cls: e.g. Adam
            gen_optimizer_args: dict of args to give to generator's optimizer
                (excluding model params, of course)
            cri_optimizer_args: dict of args to give to critic's optimizer
                (excluding model params, of course)
        """
        super().__init__()
        self.gen = generator
        self.cri = critic
        self.gen_func = generator_func
        self.wgan_funcs = WGANWithGradientPenaltyFuncs(gp_weight, critic_iterations)
        self.optimizer_cls = optimizer_cls
        self.gen_optimizer_args = gen_optimizer_args
        self.cri_optimizer_args = cri_optimizer_args

        self.automatic_optimization = False
    
    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx):
        opt_g, opt_c = self.optimizers()

        self.toggle_optimizer(opt_c)

        # Creating fake images
        with torch.no_grad():
            fake_imgs = self.gen_func(self.gen, batch)
        
        loss_critic, w_dist, grad_pen = self.wgan_funcs.critic_loss(
            self.cri, batch, fake_imgs)

        # Weight update:
        opt_c.zero_grad()
        self.manual_backward(loss_critic)
        opt_c.step()
        self.log("train/loss_critic", loss_critic, prog_bar=True)
        self.log("train/wasserstein_distance", w_dist)
        self.log("train/gradient_penalty", grad_pen)

        self.untoggle_optimizer(opt_c) # not really sure if needed but well..
        
        also_train_generator = self.wgan_funcs.also_train_generator(batch_idx)

        if also_train_generator:
            self.toggle_optimizer(opt_g)

            # Generate images:
            fake_imgs = self.gen_func(self.gen, batch)
            loss_gen = self.wgan_funcs.generator_loss(self.cri, fake_imgs)

            # Weight update:
            opt_g.zero_grad()
            self.manual_backward(loss_gen)
            opt_g.step()
            
            self.log("train/loss_generator", loss_gen, prog_bar=True)

            self.untoggle_optimizer(opt_g)
        
    def validation_step(self, batch, batch_idx):

        # Creating fake images
        fake_imgs = self.gen_func(self.gen, batch)
        
        loss_critic, w_dist, grad_pen = self.wgan_funcs.critic_loss(
            self.cri, batch, fake_imgs, False)
        loss_gen = self.wgan_funcs.generator_loss(self.cri, fake_imgs)

        # Logging:
        self.log("val/loss_critic", loss_critic, prog_bar=True)
        self.log("val/wasserstein_distance", w_dist)
        self.log("val/gradient_penalty", grad_pen)
        
        # This second one is a bit silly: won't be different under validation
        # conditions I think (unless self.gen.eval() has impact).
        self.log("val/loss_generator", loss_gen, prog_bar=True)
    
    def configure_optimizers(self):
        opt_g = self.optimizer_cls(self.gen.parameters(), **self.gen_optimizer_args)
        opt_c = self.optimizer_cls(self.cri.parameters(), **self.cri_optimizer_args)
        return opt_g, opt_c