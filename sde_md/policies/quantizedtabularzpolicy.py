import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


from typing import Callable, Tuple, Union, List

TensorDict = dict[str, torch.Tensor]
PyTorchObs = Union[torch.Tensor, TensorDict]


class QuantizedTabularZPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        grid_bins=40,
        radius=2.0,
        action_radius=1.0,
        dt=0.01,
        T=1.0,
        param_dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        # dtype of the tabular parameters; the DPP solvers request float64.
        # Must be set before super().__init__, which calls self._build().
        self.param_dtype = param_dtype
        self.grid_bins = grid_bins
        self.radius = radius  # Spatial boundary
        self.action_radius = action_radius  # Ball boundary R for actions
        self.spatial_dim = observation_space.shape[0] - 1

        self.T = T
        self.dt = dt
        self.time_bins = int(np.ceil(self.T / self.dt)) + 2
        self.total_states = (self.grid_bins**self.spatial_dim) * self.time_bins

        # The dual variable Z is 2D (same as spatial dim)
        self.action_dim = self.spatial_dim

        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        # Initialize Dual Variable Z to 0 (which maps to action 0)
        self.tabular_params = nn.ParameterDict(
            {
                "Z_table": nn.Parameter(
                    torch.zeros(self.total_states, self.action_dim, dtype=self.param_dtype)
                ),
                "value_table": nn.Parameter(
                    torch.zeros(self.total_states, 1, dtype=self.param_dtype)
                ),
            }
        )

        if hasattr(self, "mlp_extractor"):
            del self.mlp_extractor
        if hasattr(self, "action_net"):
            del self.action_net
        if hasattr(self, "value_net"):
            del self.value_net

    def _quantize(self, obs: torch.Tensor) -> torch.Tensor:
        # [Same exact quantization logic as your previous policy]
        obs = obs.clone()
        spatial_obs = obs[:, :-1]
        time_obs = obs[:, -1]

        spatial_clamped = torch.clamp(spatial_obs, -self.radius, self.radius)
        spatial_normalized = (spatial_clamped + self.radius) / (2 * self.radius)
        spatial_indices = (spatial_normalized * self.grid_bins * 0.99999).long()

        spatial_flat = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        for d in range(self.spatial_dim):
            stride = self.grid_bins ** (self.spatial_dim - 1 - d)
            spatial_flat += spatial_indices[:, d] * stride

        time_idx = torch.clamp((time_obs / self.dt).long(), 0, self.time_bins - 1)
        return spatial_flat * self.time_bins + time_idx

    def get_actions_from_Z(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Maps Z to u using the mirror map nabla psi^*(Z).
        """
        R2 = self.action_radius**2
        Z_norm_sq = torch.sum(Z**2, dim=-1, keepdim=True)
        denom = 1.0 + torch.sqrt(1.0 + R2 * Z_norm_sq)
        return (R2 * Z) / denom

    def forward(self, obs: torch.Tensor, deterministic: bool = True):
        state_indices = self._quantize(obs)
        Z = self.tabular_params["Z_table"][state_indices]
        values = self.tabular_params["value_table"][state_indices]

        # Policy is entirely deterministic in this continuous setting
        actions = self.get_actions_from_Z(Z)

        # Log probs are theoretically delta functions, return dummy for SB3 compatibility
        log_probs = torch.zeros(actions.shape[0], device=self.device)
        return actions, values, log_probs

    def get_action_Z_np(self, obs: torch.Tensor) -> np.ndarray:
        state_indices = self._quantize(obs)
        return self.tabular_params["Z_table"][state_indices].detach().cpu().numpy()

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Returns the stored value function for the given observations.
        Required for the evaluation plotters.
        """
        state_indices = self._quantize(obs)
        return self.tabular_params["value_table"][state_indices]

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = True
    ) -> torch.Tensor:
        """
        Required for SB3's policy.predict() to work.
        Maps the current observation to the continuous physical action.
        """
        state_indices = self._quantize(observation)
        Z = self.tabular_params["Z_table"][state_indices]

        # In this continuous formulation, the policy is always deterministic
        actions = self.get_actions_from_Z(Z)

        return actions
