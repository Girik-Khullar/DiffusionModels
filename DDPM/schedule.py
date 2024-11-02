import math
import torch
import matplotlib.pyplot as plt

# Defines the noising schedule for the forward process
class VarianceSchedule:
    def __init__(self, num_timesteps: int, schedule_type='linear'):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.betas = self._create_schedule()
        self.alphas = self._compute_alphas()

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
        return torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)

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
        x = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)

    def _compute_alphas(self):
        """Computes alpha values from beta values."""
        alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    def get_betas(self):
        """Returns the precomputed beta schedule."""
        return self.betas

    def get_alphas_cumprod(self):
        """Returns the cumulative product of alpha values."""
        return self.alphas


if __name__ == "__main__":
    # Example usage
    num_timesteps = 1000
    schedule_types = ['linear', 'sigmoid', 'cosine']

    plt.figure(figsize=(12, 8))

    for schedule_type in schedule_types:
        variance_schedule = VarianceSchedule(num_timesteps, schedule_type)
        alphas_cumprod = variance_schedule.get_alphas_cumprod()

        # Plotting the alpha cumulative product for each schedule type
        plt.plot(alphas_cumprod.numpy(), label=f'Cumulative Product of Alphas ({schedule_type})')

    plt.title("Cumulative Product of Alphas for Different Schedules")
    plt.xlabel("Timesteps")
    plt.ylabel("Alpha Cumulative Product")
    plt.grid(True)
    plt.legend()
    plt.show()