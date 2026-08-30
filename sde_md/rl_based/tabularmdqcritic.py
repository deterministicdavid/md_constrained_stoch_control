import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.utils import FloatSchedule, explained_variance

SelfTabularMDQcritic = TypeVar("SelfTabularMDQCritic", bound="TabularMDQCritic")

from sde_md.policies.quantizedtabularpolicy import QuantizedTabularPolicy


class TabularMDQCritic(OffPolicyAlgorithm):

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1M transitions
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        md_lambda: Union[float, Schedule] = 10,
        ent_coef: float = 0.1,  # tau in some papers
        normalize_advantage: bool = True,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,  # Discrete SAC doesn't use action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=False,
            support_multi_env=True,
            supported_action_spaces=(spaces.Discrete, spaces.MultiDiscrete),
        )

        self.md_lambda = md_lambda
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # Initialize schedule for MD step size
        self.md_lambda = FloatSchedule(self.md_lambda)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        # Update learning rates
        self._update_learning_rate(self.policy.critic.optimizer)
        if hasattr(self.policy.actor, "optimizer"):
            self._update_learning_rate(self.policy.actor.optimizer)

        md_lambda_val = self.md_lambda(self._current_progress_remaining)

        critic_losses = []
        ent_coef = self.ent_coef  # Fixed alpha for now

        # Pre-compute logit offsets for mapping actions to the flat table
        if isinstance(self.action_space, spaces.MultiDiscrete):
            nvec = self.action_space.nvec
            logit_offsets = np.cumsum(np.concatenate(([0], nvec[:-1])))
            total_logit_cols = int(np.sum(nvec))
            n_dims = len(nvec)
        else:
            logit_offsets = [0]
            total_logit_cols = int(self.action_space.n)
            n_dims = 1

        logit_offsets_tensor = th.tensor(
            logit_offsets, device=self.device, dtype=th.long
        )

        for gradient_step in range(gradient_steps):
            # 1. Sample Replay Buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # Ensure actions_long is strictly (batch_size, n_dims)
            actions_long = replay_data.actions.long()
            if len(actions_long.shape) == 1:
                actions_long = actions_long.unsqueeze(1)

            # Map environment actions to tabular column indices
            col_indices = actions_long + logit_offsets_tensor

            with th.no_grad():
                # 2. Compute Target Q-Values for next state
                next_action_probs, next_log_probs = self.policy.get_action_probs(
                    replay_data.next_observations
                )
                next_q = self.policy.critic(
                    self.policy._quantize(replay_data.next_observations)
                )

                # Properly calculate V(s') per dimension, then average for a global target
                if isinstance(self.action_space, spaces.MultiDiscrete):
                    split_p = th.split(next_action_probs, list(nvec), dim=-1)
                    split_lp = th.split(next_log_probs, list(nvec), dim=-1)
                    split_q = th.split(next_q, list(nvec), dim=-1)

                    v_per_dim = []
                    for p, lp, q in zip(split_p, split_lp, split_q):
                        v_per_dim.append(
                            (p * (q - ent_coef * lp)).sum(dim=-1, keepdim=True)
                        )
                    next_v = th.cat(v_per_dim, dim=-1).mean(dim=-1, keepdim=True)
                else:
                    next_v = (
                        next_action_probs * (next_q - ent_coef * next_log_probs)
                    ).sum(dim=-1, keepdim=True)

                target_q = (
                    replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v
                )

            # 3. Compute Current Q-Values for Critic Update
            state_indices = self.policy._quantize(replay_data.observations)
            current_q = self.policy.critic(state_indices)

            # Gather Q-values using col_indices (this pulls from the correct offset columns!)
            current_q_taken = current_q.gather(1, col_indices)

            # 4. Exact Tabular Critic Update (Q-Learning)
            with th.no_grad():
                # Compute TD Errors. Shape: (batch_size, n_dims)
                td_error = target_q.expand_as(current_q_taken) - current_q_taken

                # For logging only (calculate MSE)
                critic_loss = (td_error**2).mean()
                critic_losses.append(critic_loss.item())

                # Get the current learning rate from the schedule
                lr = self.learning_rate(self._current_progress_remaining)

                # Flatten indices to (batch_size * n_dims) to update the 1D view of the table
                flat_indices = (
                    state_indices.unsqueeze(1) * total_logit_cols + col_indices
                ).reshape(-1)

                # Count occurrences of each specific state-action index
                counts = th.bincount(flat_indices)
                batch_counts_flat = counts[
                    flat_indices
                ]  # Shape: (batch_size * n_dims,)

                # Flatten TD error to match (batch_size * n_dims,)
                td_error_flat = td_error.reshape(-1)

                # Scale update: multiply by LR, divide by exact occurrences
                q_updates_flat = (lr * td_error_flat) / batch_counts_flat

                # Apply updates directly to the Q-tables!
                self.policy.critic.q_table.view(-1).index_add_(
                    0, flat_indices, q_updates_flat
                )

            # # 5. Mirror Descent Actor Update
            with th.no_grad():
                current_action_probs, current_log_probs = self.policy.get_action_probs(
                    replay_data.observations
                )
                # q_min_all = th.min(current_q1, current_q2)

                # Calculate Advantages for ALL actions
                if isinstance(self.action_space, spaces.MultiDiscrete):
                    split_p = th.split(current_action_probs, list(nvec), dim=-1)
                    split_q = th.split(current_q, list(nvec), dim=-1)

                    adv_per_dim = []
                    for p, q in zip(split_p, split_q):
                        # V_dim shape: (batch_size, 1)
                        v_dim = (p * q).sum(dim=-1, keepdim=True)
                        # q shape: (batch_size, n_actions_in_dim), broadcasts correctly!
                        adv_per_dim.append(q - v_dim)

                    # Re-concatenate into (batch_size, total_action_cols)
                    advantages_all = th.cat(adv_per_dim, dim=-1)
                else:
                    current_v = (current_action_probs * current_q).sum(
                        dim=-1, keepdim=True
                    )
                    advantages_all = current_q - current_v

                # Exact MD target for ALL actions in the sampled states
                md_target_all = advantages_all - self.ent_coef * current_log_probs
                state_indices_flat = state_indices.long().view(-1)
                state_counts = th.bincount(state_indices_flat)
                batch_state_counts = state_counts[state_indices_flat].unsqueeze(1)
                updates_all = (md_target_all * md_lambda_val) / batch_state_counts

                # Add directly to the 2D tensor! (state_indices is already 1D)
                self.policy.actor.action_logits.index_add_(
                    0, state_indices.long(), updates_all
                )

                # Shift logits to log-probabilities (integrates to 1)
                nvec_arg = (
                    self.action_space.nvec
                    if isinstance(self.action_space, spaces.MultiDiscrete)
                    else None
                )
                self.policy.actor.normalize_policy_(nvec=nvec_arg)

            # # 5. Exact Soft Policy Iteration (Actor Update)
            # with th.no_grad():
            #     state_indices_flat = state_indices.long().view(-1)

            #     # 1. Fetch the freshly updated Q-values for the sampled states
            #     latest_q = self.policy.critic.q_table[state_indices_flat]

            #     # 2. Optimal SAC policy logits are exactly Q(s, a) / alpha
            #     optimal_logits = latest_q / self.ent_coef

            #     # 3. Directly overwrite the actor's logits (no learning rate needed!)
            #     self.policy.actor.action_logits[state_indices_flat] = optimal_logits

            #     # 4. Normalize to true log-probabilities (handles MultiDiscrete automatically)
            #     nvec_arg = self.action_space.nvec if isinstance(self.action_space, spaces.MultiDiscrete) else None
            #     self.policy.actor.normalize_policy_(nvec=nvec_arg)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/md_lambda", md_lambda_val)

    def learn(
        self: SelfTabularMDQcritic,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TabularMDQCritic",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTabularMDQcritic:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
