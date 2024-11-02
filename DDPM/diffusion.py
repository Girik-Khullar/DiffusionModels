from .schedule import VarianceSchedule
import torch
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

class Diffusion:
    def __init__(self, device, noise_steps=1000, schedule_type="linear", img_size=256):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device

        # Instantiate VarianceSchedule and get betas and alphas
        variance_schedule = VarianceSchedule(noise_steps, schedule_type)
        self.beta = variance_schedule.get_betas().to(self.device)

        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=2, p_uncound=0.1):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                # Randomly generate mask
                z_uncound = torch.rand(n)
                batch_mask = (z_uncound > p_uncound).long().to(self.device)
                # print(x.shape, t.shape, labels.shape, batch_mask.shape)
                predicted_noise = model(x, t, labels, batch_mask)

                if cfg_scale > 0:
                    batch_mask = (z_uncound > 1).int().to(self.device)
                    uncond_predicted_noise = model(x, t, labels, batch_mask)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def display_images(samples, nrow=8):
    # Check the shape of x
    print("Shape of x:", samples.shape)  # Expected shape [batch_size, 3, height, width]

    # Create a grid of images using make_grid, directly using uint8 format
    img_grid = vutils.make_grid(samples, nrow=nrow, normalize=False)  # No need to normalize

    # Display the grid of images
    plt.figure(figsize=(12, 12))
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())  # Convert to HWC format for imshow
    plt.axis('off')  # Hide axis
    plt.show()