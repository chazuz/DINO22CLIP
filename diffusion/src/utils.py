import torch
from diffusion.configs.diffusion_config import TIMESTEPS, BETA_START, BETA_END, DEVICE

class Diffusion:
    """Conditional diffusion for embeddings."""
    def __init__(self, T=TIMESTEPS, device=DEVICE):
        self.T = T
        self.device = device
        self.beta = torch.linspace(BETA_START, BETA_END, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

    @torch.no_grad()
    def sample_loop(self, model, z_cond, shape):
        x_t = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps_pred = model(x_t, t_tensor, z_cond)
            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            x_t = 1 / torch.sqrt(alpha_t) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
            if t > 0:
                x_t = x_t + torch.sqrt(beta_t) * torch.randn_like(x_t)
        return x_t
