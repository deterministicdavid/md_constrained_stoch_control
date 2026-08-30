import numpy as np
import torch


class ProjectedLQRPolicy:
    def __init__(
        self,
        T: float,
        action_radius: float,
        radius: float = 1.0, # just to match things that exist in policies which have to quantize
        control_cost: float = 0.1,
        terminal_cost: float = 2.0,
        sigma: float = 0.0,
        state_dim: int = 2,
        dt: float = 0, # just to match things that exist in policies which have to quantize
    ):

        self.T = T
        self.dt = dt
        self.action_radius = action_radius
        self.radius = radius
        self.r = control_cost
        self.qf = terminal_cost
        self.state_dim = state_dim
        self.sigma = sigma
        self.device = "cpu"

    def eval(self):
        pass  # Dummy to satisfy setup_env_and_policy

    def predict(self, obs: np.ndarray, deterministic=True):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Handle unbatched 1D observations from a single environment
        is_single = False
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
            is_single = True

        # obs is shape (N, 3) where columns are [x, y, t]
        x_vec = obs[:, :2]
        t = obs[:, 2]

        # 1. Exact Analytical LQR Solution: P(t) = 1 / (1/q_f + (T - t)/r)
        # (This is a scalar because the matrices are isotropic multiples of the Identity)
        P_t = 1.0 / ((1.0 / self.qf) + (self.T - t) / self.r)

        # 2. Optimal Unconstrained Control: u^*(t) = - R^{-1} B^T P(t) x(t)
        u_unconstrained = -(1.0 / self.r) * P_t[:, None] * x_vec

        # 3. Project onto the Admissible Action Ball
        u_norm = np.linalg.norm(u_unconstrained, axis=1, keepdims=True)
        u_norm_safe = np.where(u_norm == 0, 1.0, u_norm)  # Prevent div-by-zero

        # Scale down if norm exceeds action_radius
        scale = np.minimum(1.0, self.action_radius / u_norm_safe)
        u_projected = u_unconstrained * scale

        # Return to 1D if the input was 1D
        if is_single:
            u_projected = u_projected[0]

        return u_projected, None

    # We add predict_values so you can plot the LQR value function!
    def predict_values(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        # Handle unbatched 1D tensor inputs
        is_single = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            is_single = True

        x_vec = obs[:, :2]
        t = obs[:, 2]

        P_t = 1.0 / ((1.0 / self.qf) + (self.T - t) / self.r)
        p_t = (
            self.state_dim
            * (self.sigma**2)
            * self.r
            * torch.log(1.0 + (self.qf / self.r) * (self.T - t))
        )
        cost_to_go = P_t * torch.sum(x_vec**2, dim=1) + p_t

        V = -cost_to_go

        V_out = V.unsqueeze(1)

        # Return to 0D scalar tensor if input was 1D
        if is_single:
            V_out = V_out[0]

        return V_out

    def __call__(self, obs, deterministic=True):
        """
        Mimics the forward pass of a PyTorch nn.Module policy.
        Returns (actions_tensor, values, log_probs)
        """
        # Get actions using our existing numpy-based predict method
        actions_np, _ = self.predict(obs, deterministic=deterministic)

        # Wrap it in a tensor to satisfy the caller
        actions_tensor = torch.tensor(
            actions_np, dtype=torch.float32, device=self.device
        )

        # Return dummy values for value and log_prob
        return actions_tensor, None, None
