import torch
from diffusion.configs.diffusion_config import TIMESTEPS, BETA_START, BETA_END, DEVICE

class DiffusionDDIM:
    """Deterministic DDIM sampling for embeddings."""

    def __init__(self, T=TIMESTEPS, device=DEVICE):
        self.T = T
        self.device = device
        # Linear beta schedule
        self.beta = torch.linspace(BETA_START, BETA_END, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @torch.no_grad()
    def sample_loop(self, model, z_cond, shape):
        """
        Deterministic DDIM sampling loop.

        Args:
            model: Conditional denoiser model
            z_cond: Conditioning embeddings (e.g., CLIP or DINO)
            shape: Output shape (num_samples, embedding_dim)
        Returns:
            x_0: Samples generated from DDIM
        """


        # Initialize x_T from standard Gaussian
        x_t = torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps_pred = model(x_t, t_tensor, z_cond)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]

            # alpha_bar at previous timestep
            alpha_bar_prev = self.alpha_bar[t-1] if t > 0 else torch.tensor(1.0, device=self.device)

            # -----------------------------
            # DDIM deterministic update
            # -----------------------------
            x_t = (
                torch.sqrt(alpha_bar_prev) *
                (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
                + torch.sqrt(1 - alpha_bar_prev) * eps_pred
            )

        return x_t
