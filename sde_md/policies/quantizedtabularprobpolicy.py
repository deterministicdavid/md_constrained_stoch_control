import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from typing import Callable, Tuple, Union, List

TensorDict = dict[str, torch.Tensor]
PyTorchObs = Union[torch.Tensor, TensorDict]


class QuantizedTabularProbPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        grid_bins: int = 10,
        radius: float = 2.0,
        dt: float = 0.1,
        T: float = 1.0,
        action_params_to_optimizer: bool = False,
        **kwargs
    ):
        self.grid_bins = grid_bins
        self.radius = radius
        self.spatial_dim = observation_space.shape[0] - 1

        self.T = T
        self.dt = dt
        self.time_bins = int(np.ceil(self.T / self.dt)) + 2

        self.n_spatial_states = self.grid_bins**self.spatial_dim
        self.total_states = self.n_spatial_states * self.time_bins

        self.action_params_to_optimizer = action_params_to_optimizer

        if isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(np.sum(action_space.nvec))
        else:
            self.action_dim = int(action_space.n)

        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        # Initialize with Uniform Probabilities instead of Logits
        init_probs = torch.ones(self.total_states, self.action_dim) / self.action_dim

        self.tabular_params = nn.ParameterDict(
            {
                "action_probs": nn.Parameter(init_probs),
                "value_table": nn.Parameter(torch.zeros(self.total_states, 1)),
            }
        )

        if hasattr(self, "mlp_extractor"):
            del self.mlp_extractor
        if hasattr(self, "action_net"):
            del self.action_net
        if hasattr(self, "value_net"):
            del self.value_net

        params_to_optimize = (
            self.parameters()
            if self.action_params_to_optimizer
            else [self.tabular_params["value_table"]]
        )
        self.optimizer = self.optimizer_class(
            params_to_optimize, lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _quantize(self, obs: torch.Tensor) -> torch.Tensor:
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

    def _get_action_dist_from_probs(self, probs: torch.Tensor):
        # SB3 distributions expect logits, so we convert safely on the fly
        safe_logits = torch.log(probs + 1e-12)
        return self.action_dist.proba_distribution(action_logits=safe_logits)

    def get_distribution(self, obs: PyTorchObs):
        state_indices = self._quantize(obs)
        probs = self.tabular_params["action_probs"][state_indices]
        return self._get_action_dist_from_probs(probs)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_indices = self._quantize(obs)
        values = self.tabular_params["value_table"][state_indices]
        probs = self.tabular_params["action_probs"][state_indices]

        distribution = self._get_action_dist_from_probs(probs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_indices = self._quantize(obs)
        values = self.tabular_params["value_table"][state_indices]
        probs = self.tabular_params["action_probs"][state_indices]

        distribution = self._get_action_dist_from_probs(probs)
        return values, distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.tabular_params["value_table"][self._quantize(obs)]

    def get_action_probs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_indices = self._quantize(obs)
        probs = self.tabular_params["action_probs"][state_indices]

        if isinstance(self.action_space, spaces.MultiDiscrete):
            split_probs = torch.split(probs, list(self.action_space.nvec), dim=-1)
            log_probs = torch.cat([torch.log(p + 1e-12) for p in split_probs], dim=-1)
            return probs, log_probs
        return probs, torch.log(probs + 1e-12)


    def _project_simplex_batched(self, v: torch.Tensor) -> torch.Tensor:
        """
        Projects a 2D tensor (B, D) onto the probability simplex along the last dimension.
        Fully vectorized PyTorch equivalent of the O(n log n) sorting algorithm.
        """
        D = v.shape[-1]
        
        # 1. Sort descending along the action dimension
        u, _ = torch.sort(v, descending=True, dim=-1)
        
        # 2. Cumulative sum
        cssv = torch.cumsum(u, dim=-1)
        
        # 3. Create the index vector (1 to D)
        idx = torch.arange(1, D + 1, dtype=v.dtype, device=v.device)
        
        # 4. Find the condition
        cond = (u * idx) > (cssv - 1.0)
        
        # 5. Find rho. Since cond is monotonically decreasing (True...True...False...False),
        # the number of True values minus 1 gives us the index of the last True value.
        rho = cond.sum(dim=-1, keepdim=True) - 1
        
        # 6. Gather the specific cssv value at rho
        cssv_rho = torch.gather(cssv, -1, rho)
        
        # 7. Calculate theta and apply it
        theta = (cssv_rho - 1.0) / (rho + 1.0)
        w = torch.clamp(v - theta, min=0.0)
        
        return w

    def normalize_policy_dnu_(self, nvec: Union[List[int], np.ndarray] = None) -> None:
        """
        In-place flooring and normalization of probabilities (integrates to 1).
        """
        with torch.no_grad():
            probs_param = self.tabular_params["action_probs"]

            # 1. Floor at 0
            probs_param.clamp_(min=0.0)

            # 2. Normalize
            if nvec is not None:
                split_probs = torch.split(probs_param, list(nvec), dim=-1)
                norm_probs = [
                    p / (p.sum(dim=-1, keepdim=True) + 1e-12) for p in split_probs
                ]
                probs_param.copy_(torch.cat(norm_probs, dim=-1))
            else:
                probs_param.copy_(
                    probs_param / (probs_param.sum(dim=-1, keepdim=True) + 1e-12)
                )

    def normalize_policy_(self, nvec: Union[List[int], np.ndarray] = None) -> None:
        """
        In-place projection of parameters onto the probability simplex.
        """
        with torch.no_grad():
            probs_param = self.tabular_params["action_probs"]

            if nvec is not None:
                # MultiDiscrete: split, project each action dimension independently, then concatenate
                split_probs = torch.split(probs_param, list(nvec), dim=-1)
                projected_probs = [
                    self._project_simplex_batched(p) for p in split_probs
                ]
                probs_param.copy_(torch.cat(projected_probs, dim=-1))
            else:
                # Discrete: project directly across the entire row
                probs_param.copy_(self._project_simplex_batched(probs_param))