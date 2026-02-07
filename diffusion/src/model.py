import torch
import torch.nn as nn

class ConditionalDenoiser(nn.Module):
    """MLP conditional denoiser for embeddings."""
    def __init__(self, target_dim, cond_dim, hidden_dim=1024, n_hidden=2):
        super().__init__()
        self.input_dim = target_dim + cond_dim + 1  # +1 for timestep
        layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t, cond):
        t_norm = t.float().unsqueeze(-1) / 1000.0
        x = torch.cat([z_t.float(), cond.float(), t_norm], dim=-1)
        return self.net(x)
