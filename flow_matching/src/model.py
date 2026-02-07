import torch
import torch.nn as nn


class ConditionalFlowModel(nn.Module):
    """
    MLP conditional velocity predictor for flow matching.
    
    Architecture mirrors ConditionalDenoiser from diffusion/src/model.py,
    but predicts velocity field v(x_t, t, cond) instead of noise.
    """
    
    def __init__(self, target_dim, cond_dim, hidden_dim=1024, n_hidden=2):
        super().__init__()
        self.input_dim = target_dim + cond_dim + 1  # +1 for timestep
        
        layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, target_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x_t, t, cond):
        """
        Args:
            x_t: (batch, target_dim) - current state at time t
            t: (batch,) - continuous time in [0, 1]
            cond: (batch, cond_dim) - conditioning embedding
        
        Returns:
            v_t: (batch, target_dim) - predicted velocity
        """
        t_norm = t.float().unsqueeze(-1)  # Already in [0, 1], just add dimension
        x = torch.cat([x_t.float(), cond.float(), t_norm], dim=-1)
        return self.net(x)
