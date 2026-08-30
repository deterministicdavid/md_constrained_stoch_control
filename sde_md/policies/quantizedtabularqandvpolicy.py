import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from typing import Callable, Tuple, Union

TensorDict = dict[str, torch.Tensor]
PyTorchObs = Union[torch.Tensor, TensorDict]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import make_proba_distribution

from typing import Callable, Tuple, Union

from sde_md.policies.quantizedtabularqpolicy import TabularActor, TabularCritic

TensorDict = dict[str, torch.Tensor]
PyTorchObs = Union[torch.Tensor, TensorDict]


class TabularValueCritic(nn.Module):
    def __init__(self, total_states: int):
        super().__init__()
        # V(s) is just a single value per state
        self.v_table = nn.Parameter(torch.zeros(total_states, 1))

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        return self.v_table[state_indices]


class QuantizedTabularQandVPolicy(BasePolicy):
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

        self.action_logits_to_optimizer = action_logits_to_optimizer

        # This implementation assumes a standard Discrete action space for the Q-table shapes.
        if isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(np.sum(action_space.nvec))
        else:
            self.action_dim = int(action_space.n)

        self.action_dist = make_proba_distribution(action_space)

        # BasePolicy requires an optimizer_class and kwargs, usually handled by super
        super().__init__(observation_space, action_space, **kwargs)
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = TabularActor(self.total_states, self.action_dim)
        nvec_arg = (
            self.action_space.nvec
            if isinstance(self.action_space, spaces.MultiDiscrete)
            else None
        )
        self.actor.normalize_policy_(nvec=nvec_arg)

        self.critic = TabularCritic(self.total_states, self.action_dim)
        self.v_critic = TabularValueCritic(self.total_states)  # <-- NEW

        # Standalone dummy parameters
        dummy_critic_param = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.critic.optimizer = self.optimizer_class([dummy_critic_param], lr=0.0)

        dummy_v_param = nn.Parameter(torch.tensor([0.0], requires_grad=True))  # <-- NEW
        self.v_critic.optimizer = self.optimizer_class(
            [dummy_v_param], lr=0.0
        )  # <-- NEW

        if self.action_logits_to_optimizer:
            self.actor.optimizer = self.optimizer_class(
                self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
        else:
            dummy_actor_param = nn.Parameter(torch.tensor([0.0], requires_grad=True))
            self.actor.optimizer = self.optimizer_class([dummy_actor_param], lr=0.0)

    def _quantize(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Preserved exact quantization logic.
        """
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

        time_idx = (time_obs / self.dt).long()
        time_idx = torch.clamp(time_idx, 0, self.time_bins - 1)

        final_flat_indices = spatial_flat * self.time_bins + time_idx

        return final_flat_indices

    def get_action_probs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_indices = self._quantize(obs)
        logits = self.actor(state_indices)

        if isinstance(self.action_space, spaces.MultiDiscrete):
            # Split logits per action dimension and apply softmax per dimension
            nvec = self.action_space.nvec
            split_logits = torch.split(logits, list(nvec), dim=-1)

            action_probs_list = [F.softmax(l, dim=-1) for l in split_logits]
            log_probs_list = [F.log_softmax(l, dim=-1) for l in split_logits]

            action_probs = torch.cat(action_probs_list, dim=-1)
            log_probs = torch.cat(log_probs_list, dim=-1)
        else:
            action_probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

        return action_probs, log_probs

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Required by BasePolicy for environment stepping.
        """
        state_indices = self._quantize(observation)
        logits = self.actor(state_indices)

        distribution = self.action_dist.proba_distribution(action_logits=logits)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Directly query the explicitly learned Value function.
        """
        state_indices = self._quantize(obs)
        return self.v_critic(state_indices)
