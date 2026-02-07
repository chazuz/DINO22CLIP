import torch
from flow_matching.configs.flow_config import NUM_INTEGRATION_STEPS, SIGMA_MIN, DEVICE


class FlowMatching:
    """
    Conditional flow matching for embeddings.
    Mirrors the Diffusion class structure from diffusion/src/utils.py
    """
    
    def __init__(self, num_steps=NUM_INTEGRATION_STEPS, sigma_min=SIGMA_MIN, device=DEVICE):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.device = device
        self.dt = 1.0 / num_steps
    
    def sample_trajectory(self, x_1, x_0=None):
        """
        Sample a point on the interpolation trajectory for training.
        
        Args:
            x_1: (batch, dim) - target embedding (data)
            x_0: (batch, dim) - source noise (if None, sample from N(0, I))
        
        Returns:
            x_t: (batch, dim) - interpolated point at time t
            t: (batch,) - sampled time in [0, 1]
            u_t: (batch, dim) - true velocity (x_1 - x_0)
        """
        batch_size = x_1.shape[0]
        
        # Sample x_0 from standard normal if not provided
        if x_0 is None:
            x_0 = torch.randn_like(x_1, device=self.device)
        
        # Sample time uniformly from [0, 1]
        t = torch.rand(batch_size, device=self.device)
        
        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        t_expanded = t.view(-1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # True velocity for linear interpolation: u_t = x_1 - x_0
        u_t = x_1 - x_0
        
        return x_t, t, u_t
    
    @torch.no_grad()
    def sample_loop(self, model, z_cond, shape):
        """
        Generate samples by integrating the ODE using Euler method.
        Mirrors the sample_loop signature from Diffusion class.
        
        Args:
            model: Trained ConditionalFlowModel
            z_cond: (batch, cond_dim) - conditioning embeddings
            shape: (batch_size, target_dim) - shape of output
        
        Returns:
            x_1: (batch, target_dim) - generated embeddings at t=1
        """
        batch_size = shape[0]
        
        # Start from noise at t=0
        x_t = torch.randn(shape, device=self.device)
        
        # Integrate ODE from t=0 to t=1 using Euler method
        for step in range(self.num_steps):
            t = torch.full((batch_size,), step * self.dt, device=self.device)
            v_t = model(x_t, t, z_cond)
            x_t = x_t + self.dt * v_t
        
        return x_t
