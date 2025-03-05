
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch import nn
from PIL import Image
from torch.nn import functional as F
from pytorch_fid import fid_score

class Diffusion():
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepare_noise_schedule(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        return betas

    def add_noise(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return x * sqrt_alpha_hat + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(0, self.noise_steps, (n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x- ((1 - alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x*255).type(torch.uint8)
        return x

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)
        emb = emb[:x.shape[0], :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)
        emb = emb[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class BigUnet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
def get_data(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataloader

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1,2,0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def train(args):
    device = args.device
    dataloader = get_data(args)
    model = BigUnet().to(device)
    discriminator = Discriminator().to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCELoss()
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.lr)
    diffusion = Diffusion(img_size=args.image_size, device=device)
    l = len(dataloader)

    print(model)
    print(sum(p.numel() for p in model.parameters()))

    loss_plot = []
    loss_discriminator_plot = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        pbar = tqdm(dataloader)

        #train discriminator every other epoch
        if epoch % 2 == 1:
            print("Training Discriminator")
            discriminator.train()
            for i, (images, _) in enumerate(pbar):
                images = images.to(device)
                t = diffusion.sample_timesteps(args.batch_size).to(device)

                #this are the noised images
                x_t, noise = diffusion.add_noise(images, t)

                # this is the noised images with the noise removed; the "fake" data
                predicted_noise = model(x_t, t)

                #train discriminator
                discriminator_optimizer.zero_grad()
                real = discriminator(noise)
                fake = discriminator(predicted_noise.detach())
                loss_discriminator = bce(real, torch.ones_like(real)) + bce(fake, torch.zeros_like(fake))
                loss_discriminator.backward()
                discriminator_optimizer.step()

            loss_discriminator_plot.append(loss_discriminator.item())
        
        else:
            print("Training Generator")
            discriminator.eval()
            for i, (images, _) in enumerate(pbar):
                images = images.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.add_noise(images, t)

                #predicted noise
                predicted_noise = model(x_t, t)

                #see if the discriminator can tell the difference between the real and fake images
                predicted_noise_det = predicted_noise.detach()
                discriminator_result = discriminator(x_t - predicted_noise_det)
                
                loss_gen = mse(predicted_noise, noise)
                loss_disc = 0.01 * bce(discriminator_result, torch.ones_like(discriminator_result))
                
                diffusion_loss = loss_gen + loss_disc

                # Update generator (ensure discriminator does not modify needed tensors)
                optimizer.zero_grad()
                diffusion_loss.backward() 
                optimizer.step()

            loss_plot.append(diffusion_loss.item())


        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, f'./sampled_images_{epoch}.png')
        torch.save(model.state_dict(), f'./diff_model_{epoch}.pt')
        torch.save(discriminator.state_dict(), f'./disc_model_{epoch}.pt')

        plt.plot(loss_discriminator_plot)
        plt.plot(loss_plot)
        plt.title('Loss diffusion')
        plt.savefig(f'./loss_plot.png')
        plt.close()

        plt.plot(loss_discriminator_plot)
        plt.title('Loss discriminator')
        plt.savefig(f'./loss_discriminator_plot.png')
        plt.close()
      

class Discriminator(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x).squeeze())  # Apply sigmoid

def save_images_individual(images, path, **kwargs):
    for i in range(images.shape[0]):
        im = Image.fromarray(images[i].permute(1, 2, 0).to('cpu').numpy())
        im.save(f"{path}_{i}.png")
        

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 150
    args.batch_size = 8
    args.image_size = 64
    args.dataset_path = r"./data/cifar-10-batches-py"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

def load_and_sample(epoch=150):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 150
    args.batch_size = 8
    args.image_size = 64
    args.dataset_path = r"./data/cifar-10-batches-py"
    args.device = "cuda"
    args.lr = 3e-4
    diffusion = Diffusion(img_size=args.image_size, device=args.device)
    model = BigUnet().to(args.device)
    model.load_state_dict(torch.load(f'model_{epoch}.pt'))
    sampled_images = diffusion.sample(model, n=50)
    save_images_individual(sampled_images, f'./sampled_large/sample_', format='png')

def save_cifar_as_images(n=100):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor(),

    ])

    dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for i, (images, _) in enumerate(dataloader):
        #convert images to uint8
        images = (images*255).type(torch.uint8)
        save_images_individual(images, f'./real_data/sample_{i}', format='png')
        if i == 100:
            break

def calculate_fid(path_to_images, path_to_real_images):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 150
    args.batch_size = 8
    args.image_size = 64
    args.dataset_path = r"./data/cifar-10-batches-py"
    args.device = "cuda"
    args.lr = 3e-4
    fid_value = fid_score.calculate_fid_given_paths([path_to_images, path_to_real_images], 50, args.device, 2048)
    return fid_value

def plot_FID_over_time():

    fid_values = []
    save_cifar_as_images()

    for i in range(70):
        load_and_sample(i)
        val = calculate_fid(f'./sampled_large', './real_data')
        fid_values.append(val)
        print(val)

    plt.plot(fid_values)
    plt.title('FID over time')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.savefig(f'./fid_plot.png')
    plt.close()

def main():  
    plot_FID_over_time()

    # launch()
    # save_cifar_as_images()
    # load_and_sample()
    # calculate_fid('./sample_large', './real_data')
    
if __name__ == '__main__':
    main()