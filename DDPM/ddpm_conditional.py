import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from .utils import *
from .modules import UnetModel, EMA
import torchvision.utils as vutils
import logging
from .diffusion import Diffusion, display_images
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def create_model_directory(run_name):
    """Create the model directory if it doesn't exist."""
    model_dir = os.path.join("models", run_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def load_checkpoint(model, optimizer, ema_model, model_dir):
    """Load model and optimizer states from checkpoint."""
    checkpoint_path = os.path.join(model_dir, "ckpt.pt")
    optimizer_path = os.path.join(model_dir, "optim.pt")
    ema_path = os.path.join(model_dir, "ema_ckpt.pt")

    if os.path.exists(checkpoint_path) and os.path.exists(optimizer_path):
        model.load_state_dict(torch.load(checkpoint_path))
        optimizer.load_state_dict(torch.load(optimizer_path))
        if os.path.exists(ema_path):
            ema_model.load_state_dict(torch.load(ema_path))
        print("Loaded model and optimizer from checkpoint.")
        return True  # Indicate that a checkpoint was loaded
    else:
        print("No checkpoint found, starting fresh.")
        return False  # No checkpoint loaded


def train_one_epoch(model, dataloader, optimizer, diffusion, device, epoch, args, ema, loss_fn, save_images_fn):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)

    # Track the total loss for averaging
    total_loss = 0
    num_batches = len(dataloader)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)

        # Randomly generate mask
        z_uncound = torch.rand(images.shape[0])
        batch_mask = (z_uncound > 0.10).int().to(device)

        predicted_noise = model(x_t, t, labels, batch_mask)
        loss = loss_fn(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA
        ema.step_ema(model)

        # Accumulate loss for averaging
        total_loss += loss.item()

        # Update the progress bar with the current loss
        pbar.set_postfix(MSE=loss.item())

    # Calculate and print average loss after each epoch
    avg_loss = total_loss / num_batches
    print(f"Average loss after epoch {epoch}: {avg_loss:.4f}")

    # Save generated images and model checkpoints based on args.save_epochs
    if (epoch + 1) % args.save_epochs == 0:
        epoch_str = str(epoch + 1).zfill(4)  # Zero-pad epoch number
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch_str}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, f"optim_epoch_{epoch_str}.pt"))
        torch.save(ema.state_dict(), os.path.join(model_dir, f"ema_model_epoch_{epoch_str}.pt"))

        # Generate and save images
        labels = torch.arange(10).long().to(device)
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        save_images_fn(sampled_images, os.path.join(model_dir, f"sampled_images_epoch_{epoch_str}.jpg"))


def save_model_and_results(model, optimizer, ema_model, model_dir, epoch, diffusion, device):
    """Save model, optimizer states, EMA model, and sample images."""
    torch.save(model.state_dict(), os.path.join(model_dir, "ckpt.pt"))
    torch.save(optimizer.state_dict(), os.path.join(model_dir, "optim.pt"))
    torch.save(ema_model.state_dict(), os.path.join(model_dir, "ema_ckpt.pt"))  # Save EMA model as well

    # Sample images using the diffusion model
    labels = torch.arange(10).long().to(device)  # Assuming you want to sample 10 images
    sampled_images = diffusion.sample(model, n=len(labels), labels=labels)

    # Save the sampled images
    save_image_path = os.path.join(model_dir, f"sampled_images_epoch_{epoch + 1}.png")
    vutils.save_image(sampled_images, save_image_path, nrow=5, normalize=True)
    print(f"Saved sampled images at {save_image_path}")

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UnetModel(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


# def launch():
#     import argparse
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.run_name = "DDPM_conditional"
#     args.epochs = 300
#     args.batch_size = 14
#     args.image_size = 64
#     args.num_classes = 10
#     args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
#     args.device = "cuda"
#     args.lr = 3e-4
#     train(args)


# if __name__ == '__main__':
#     launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
