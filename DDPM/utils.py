import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from schedule import VarianceSchedule


def apply_schedules(dataloader: torch.utils.data.DataLoader, n_images: int, num_timesteps: int, num_intermediate_steps: int, schedule_type='linear'):
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

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)