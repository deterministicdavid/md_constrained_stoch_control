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


class QuantizedTabularPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        grid_bins: int = 10,
        radius: float = 2.0,
        dt: float = 0.1,
        T: float = 1.0,
        action_logits_to_optimizer: bool = False,
        param_dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        # dtype of the tabular parameters. float32 (default) is what the RL code uses;
        # the DPP-based solvers (SPI, exact MD) can request float64 for higher precision.
        # Must be set before super().__init__, which calls self._build().
        self.param_dtype = param_dtype
        self.grid_bins = grid_bins
        self.radius = radius
        self.spatial_dim = observation_space.shape[0] - 1

        self.T = T
        self.dt = dt
        self.time_bins = int(np.ceil(self.T / self.dt)) + 2

        self.n_spatial_states = self.grid_bins**self.spatial_dim
        self.total_states = self.n_spatial_states * self.time_bins

        self.action_logits_to_optimizer = action_logits_to_optimizer

        if isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(np.sum(action_space.nvec))
        else:
            self.action_dim = int(action_space.n)

        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        This is where we replace the default Actor with MaskedActor.
        """
        self.tabular_params = nn.ParameterDict(
            {
                # randn(...) * 0.0 kept (rather than zeros) so the global RNG stream,
                # and hence seeded RL runs, are unchanged for the default float32.
                "action_logits": nn.Parameter(
                    torch.randn(
                        self.total_states, self.action_dim, dtype=self.param_dtype
                    )
                    * 0.0
                ),
                "value_table": nn.Parameter(
                    torch.randn(self.total_states, 1, dtype=self.param_dtype) * 0.0
                ),
            }
        )

        if hasattr(self, "mlp_extractor"):
            del self.mlp_extractor
        if hasattr(self, "action_net"):
            del self.action_net
        if hasattr(self, "value_net"):
            del self.value_net

        if self.action_logits_to_optimizer:
            # include all parameters including value function and action logits
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
        else:
            # exclude action logits from optimizer for algos that update these otherwise
            self.optimizer = self.optimizer_class(
                [self.tabular_params["value_table"]],
                lr=lr_schedule(1),
                **self.optimizer_kwargs
            )

    def _quantize(self, obs: torch.Tensor) -> torch.Tensor:
        # Clone to ensure no in-place modification of observations
        obs = obs.clone()

        # 1. Split Spatial and Time
        spatial_obs = obs[:, :-1]  # All columns except last
        time_obs = obs[:, -1]  # Last column

        spatial_clamped = torch.clamp(spatial_obs, -self.radius, self.radius)
        spatial_normalized = (spatial_clamped + self.radius) / (2 * self.radius)
        spatial_indices = (spatial_normalized * self.grid_bins * 0.99999).long()

        spatial_flat = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        for d in range(self.spatial_dim):
            stride = self.grid_bins ** (self.spatial_dim - 1 - d)
            spatial_flat += spatial_indices[:, d] * stride

        time_idx = (time_obs / self.dt).long()
        time_idx = torch.clamp(time_idx, 0, self.time_bins - 1)

        final_flat_indices = spatial_flat * self.time_bins + time_idx

        return final_flat_indices

    def _get_action_dist_from_logits(self, logits: torch.Tensor):
        """
        Helper to create the distribution object required by SB3.
        """
        return self.action_dist.proba_distribution(action_logits=logits)

    def get_distribution(self, obs: PyTorchObs):
        """
        Required by PPO to get action probabilities from observations.
        Overrides the parent method to use our table instead of a neural net.
        """
        state_indices = self._quantize(obs)
        logits = self.tabular_params["action_logits"][state_indices]
        return self.action_dist.proba_distribution(action_logits=logits)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Required for model.predict().
        """
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_indices = self._quantize(obs)

        # Look up in the dictionary
        values = self.tabular_params["value_table"][state_indices]
        logits = self.tabular_params["action_logits"][state_indices]

        # Standard distribution logic
        distribution = self._get_action_dist_from_logits(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_indices = self._quantize(obs)
        values = self.tabular_params["value_table"][state_indices]
        logits = self.tabular_params["action_logits"][state_indices]

        distribution = self._get_action_dist_from_logits(logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_probs, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        state_indices = self._quantize(obs)
        return self.tabular_params["value_table"][state_indices]

    def get_action_probs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the action probabilities and log probabilities for the given observations.
        For MultiDiscrete action spaces, it concatenates the probabilities across dimensions.
        """
        state_indices = self._quantize(obs)
        logits = self.tabular_params["action_logits"][state_indices]

        distribution = self._get_action_dist_from_logits(logits)

        # SB3's MultiCategoricalDistribution stores a list of PyTorch Categorical distributions
        if isinstance(self.action_space, spaces.MultiDiscrete):
            # Extract and concatenate probabilities for X and Y
            probs = torch.cat(
                [dist.probs for dist in distribution.distribution], dim=-1
            )
            # PyTorch's Categorical.logits actually stores the log-probabilities (log-softmax)
            log_probs = torch.cat(
                [dist.logits for dist in distribution.distribution], dim=-1
            )
        else:
            probs = distribution.distribution.probs
            log_probs = distribution.distribution.logits

        return probs, log_probs

    def normalize_policy_(self, nvec: Union[List[int], np.ndarray] = None) -> None:
        """
        In-place normalization of logits to log-probabilities (integrates to 1).
        Supports both Discrete (nvec=None) and MultiDiscrete action spaces.
        """
        with torch.no_grad():
            logits_param = self.tabular_params["action_logits"]

            if nvec is not None:
                # MultiDiscrete: split, normalize per dimension, then concatenate
                split_logits = torch.split(logits_param, list(nvec), dim=-1)
                norm_logits = [F.log_softmax(l, dim=-1) for l in split_logits]
                logits_param.copy_(torch.cat(norm_logits, dim=-1))
            else:
                # Discrete: normalize directly across the entire row
                logits_param.copy_(F.log_softmax(logits_param, dim=-1))
