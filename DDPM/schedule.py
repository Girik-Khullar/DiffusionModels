import math
import torch

# Defines the noising schedule for the forward process
class VarianceSchedule:
    def __init__(self, num_timesteps, schedule_type='linear'):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.betas = self._create_schedule()

    def _create_schedule(self):
        """Creates a beta (variance) schedule based on the chosen type."""
        if self.schedule_type == 'linear':
            return self._linear_beta_schedule()
        elif self.schedule_type == 'sigmoid':
            return self._sigmoid_beta_schedule()
        elif self.schedule_type == 'cosine':
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _linear_beta_schedule(self):
        """Linear schedule: linearly increasing beta values."""
        scale = 1000 / self.num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float64)

    def _sigmoid_beta_schedule(self):
        """Sigmoid schedule: betas follow a sigmoid curve."""
        betas = torch.linspace(-6, 6, self.num_timesteps)
        betas = torch.sigmoid(betas) / (betas.max() - betas.min()) * (0.02 - betas.min()) / 10
        return betas

    def _cosine_beta_schedule(self, s=0.008):
        """
        Cosine schedule: alphas follow a cosine curve.
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)

    def get_betas(self):
        """Returns the precomputed beta schedule."""
        return self.betas


def apply_schedules(dataloader, n_images, num_timesteps, num_intermediate_steps, schedule_type='linear'):
    # Initialize the VarianceSchedule class for the selected schedule type
    variance_schedule = VarianceSchedule(num_timesteps, schedule_type=schedule_type)
    betas = variance_schedule.get_betas()

    # Generate alpha and alpha_cumprod for diffusion steps
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    # Define intermediate timesteps to display
    step_indices = torch.linspace(0, num_timesteps - 1, num_intermediate_steps, dtype=torch.long)

    # Select a batch of images from the dataloader and limit to n_images
    images, _ = next(iter(dataloader))
    images = images[:n_images]  # Take the first n_images

    # Set up plot for multiple images and their noisy progressions
    fig, axes = plt.subplots(n_images, num_intermediate_steps + 1, figsize=(15, 5 * n_images))
    fig.suptitle(f"Noise Progression with {schedule_type.capitalize()} Schedule", fontsize=16)

    for img_idx in range(n_images):
        image = images[img_idx].unsqueeze(0)  # Select each image individually and add batch dimension

        # Display the original image
        axes[img_idx, 0].imshow(image[0].permute(1, 2, 0))
        axes[img_idx, 0].set_title("Original")
        axes[img_idx, 0].axis("off")

        # Display progressively noisier versions of the image
        for i, t in enumerate(step_indices, start=1):
            # Apply noise at step t based on the cumulative alpha product
            noise = torch.randn_like(image)
            noisy_image = (alpha_cumprod[t].sqrt() * image) + ((1 - alpha_cumprod[t]).sqrt() * noise)

            # Show the noisy image at this intermediate step
            axes[img_idx, i].imshow(noisy_image[0].permute(1, 2, 0).clamp(0, 1))
            axes[img_idx, i].set_title(f"Step {t.item()}")
            axes[img_idx, i].axis("off")

    plt.tight_layout()
    plt.show()