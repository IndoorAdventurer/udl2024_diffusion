import torch

class WGANWithGradientPenaltyFuncs:
    """Encapsulates all computations specific to the Wasserstein GAN with
    Gradient Penalty instead of weight clipping.
    """

    def __init__(self, gp_weight: float = 10, critic_iterations: int = 5):
        """
        Args:
            gp_weight: scale factor for gradient penalty term in critic loss.
            critic_iterations: this many critic learning steps for every one
                generator learning step. -1 Means never called at all.
        """
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
    
    def also_train_generator(self, batch_idx):
        """Given the batch index, checks if it is time to train the generator
        again."""
        if self.critic_iterations == -1:
            return False
        return (batch_idx % self.critic_iterations) == 0
    
    def critic_loss(self, critic_model, batch, fake_imgs, gp: bool = True):
        """Returns loss for the critic model. (gp: if False, no gradient penalty)"""
        fake_imgs = fake_imgs.detach() # just to be sure :-p

        critic_real = critic_model(batch)
        critic_fake = critic_model(fake_imgs)
        grad_pen = (
            self.compute_gradient_penalty(critic_model, batch, fake_imgs) if gp
            else 0.0
        )

        # Calculating the Wasserstein loss:
        return (
            - torch.mean(critic_real)
            + torch.mean(critic_fake)
            + self.gp_weight * grad_pen
        )
    
    def generator_loss(self, critic_model, fake_imgs):
        """Returns loss for the generator. Should only be called when
        `also_train_generator()` returns `True`
        """
        critic_fake = critic_model(fake_imgs)
        return -torch.mean(critic_fake)

    def compute_gradient_penalty(self, critic_model, real, fake):
        # Get critic outputs on random mix of real and fake images
        alpha = torch.rand(real.shape[0], *([1] * (len(real.shape) - 1)),
                           device=real.device)
        mixed = alpha * real + (1 - alpha) * fake
        mixed.requires_grad_(True)
        critic_mixed = critic_model(mixed)

        # Compute gradients of 
        gradients = torch.autograd.grad(
            inputs=mixed,
            outputs=critic_mixed,
            grad_outputs=torch.ones_like(critic_mixed),
            create_graph=True, retain_graph=True
        )[0].view(real.shape[0], -1)

        # Returning the gradient penalty:
        return (torch.mean(gradients.norm(2, dim=1) - 1) ** 2)