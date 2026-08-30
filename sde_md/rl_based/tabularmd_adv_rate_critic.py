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

SelfTabularMDAdvRateCritic = TypeVar(
    "SelfTabularMDAdvRateCritic", bound="TabularMDAdvRateCritic"
)


class TabularMDAdvRateCritic(OffPolicyAlgorithm):

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
            action_noise=None,  # Discrete doesn't use action noise
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
        dt = self.env.get_attr("dt")[0]

        q_critic_losses = []
        v_v_tgt_gaps = []
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
                state_indices = self.policy._quantize(replay_data.observations)
                next_state_indices = self.policy._quantize(
                    replay_data.next_observations
                )

                # Fetch current network evaluations
                current_action_probs, current_log_probs = self.policy.get_action_probs(
                    replay_data.observations
                )
                current_q = self.policy.critic(state_indices)
                current_q_taken = current_q.gather(1, col_indices)
                current_q_taken_joint = current_q_taken.sum(dim=-1, keepdim=True)

                current_v = self.policy.v_critic(state_indices)
                # next_v = self.policy.v_critic(next_state_indices)
                next_v = self.policy.v_critic_target(next_state_indices)

                # # --- 1. Compute Targets ---
                # # V target: Exact expectation over the CURRENT policy and CURRENT Q-table
                # if isinstance(self.action_space, spaces.MultiDiscrete):
                #     split_p = th.split(current_action_probs, list(nvec), dim=-1)
                #     split_lp = th.split(current_log_probs, list(nvec), dim=-1)
                #     split_q = th.split(current_q, list(nvec), dim=-1)

                #     v_per_dim = []
                #     for p, lp, q in zip(split_p, split_lp, split_q):
                #         v_per_dim.append((p * (q - ent_coef * lp)).sum(dim=-1, keepdim=True))
                #     target_v = th.cat(v_per_dim, dim=-1).mean(dim=-1, keepdim=True)
                # else:
                #     target_v = (current_action_probs * (current_q - ent_coef * current_log_probs)).sum(dim=-1, keepdim=True)

                # --- 1. Compute Martingale TD Error (L=1 case) ---
                # Based on the paper's Eq 4.3 loss:
                # ( V(x) - [r - q(x,a)]*dt - gamma * V_tgt(x') )^2
                martingale_td_error = (
                    current_v
                    - (replay_data.rewards - current_q_taken_joint) * dt
                    - (1 - replay_data.dones) * self.gamma * next_v
                )

                # Q target: r + gamma * next_v
                target_q = (
                    replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_v
                )

            lr = self.learning_rate(self._current_progress_remaining)
            state_indices_flat = state_indices.long().view(-1)
            flat_sa_indices = (
                state_indices.unsqueeze(1) * total_logit_cols + col_indices
            ).reshape(-1)

            # --- 2. Tabular V-Function Update ---
            with th.no_grad():
                # v_td_error = target_v - current_v
                # v_critic_losses.append((v_td_error ** 2).mean().item())

                # # Count state occurrences
                # state_counts = th.bincount(state_indices_flat)
                # batch_state_counts = state_counts[state_indices_flat].unsqueeze(1)

                # v_updates = (lr * v_td_error) / batch_state_counts

                state_counts = th.bincount(state_indices_flat)
                batch_state_counts = state_counts[state_indices_flat].unsqueeze(1)

                # Gradient descent step: V <- V - lr * Err
                v_updates = -(lr * martingale_td_error) / batch_state_counts
                self.policy.v_critic.v_table.view(-1).index_add_(
                    0, state_indices_flat, v_updates.view(-1)
                )

            # --- 3. Tabular Q-Function Update ---
            with th.no_grad():
                q_critic_losses.append((martingale_td_error**2).mean().item())

                sa_counts = th.bincount(flat_sa_indices)
                batch_sa_counts = sa_counts[flat_sa_indices]
                expanded_td_error = martingale_td_error.expand_as(
                    current_q_taken
                ).reshape(-1)

                # Gradient descent step: q <- q - lr * Err * dt
                q_updates = -(lr * expanded_td_error * dt) / batch_sa_counts

                self.policy.critic.q_table.view(-1).index_add_(
                    0, flat_sa_indices, q_updates
                )

            # 5. Mirror Descent Actor Update
            with th.no_grad():
                # current_log_probs was fetched at the start of the step
                _, current_log_probs = self.policy.get_action_probs(
                    replay_data.observations
                )

                current_q = self.policy.critic(state_indices)
                current_v = self.policy.v_critic(state_indices)

                first_variation = (
                    current_q - current_v - self.ent_coef * current_log_probs
                )

                state_indices_flat = state_indices.long().view(-1)
                state_counts = th.bincount(state_indices_flat)
                batch_state_counts = state_counts[state_indices_flat].unsqueeze(1)

                updates_all = (first_variation * md_lambda_val) / batch_state_counts

                # Add directly to the 2D tensor! (state_indices is already 1D)
                self.policy.actor.action_logits.index_add_(
                    0, state_indices_flat, updates_all
                )

                # Shift logits to log-probabilities (integrates to 1)
                nvec_arg = (
                    self.action_space.nvec
                    if isinstance(self.action_space, spaces.MultiDiscrete)
                    else None
                )
                self.policy.actor.normalize_policy_(nvec=nvec_arg)

            polyak_update(
                self.policy.v_critic.parameters(),
                self.policy.v_critic_target.parameters(),
                self.tau,
            )
            v_v_tgt_gap = (
                (
                    (
                        self.policy.v_critic(state_indices)
                        - self.policy.v_critic_target(state_indices)
                    )
                    ** 2
                )
                .mean()
                .item()
            )
            v_v_tgt_gaps.append(v_v_tgt_gap)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/q_critic_loss", np.mean(q_critic_losses))
        self.logger.record("train/v_v_tgt_gaps", np.mean(v_v_tgt_gaps))
        self.logger.record("train/md_lambda", md_lambda_val)

    def learn(
        self: SelfTabularMDAdvRateCritic,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TabularMDAdvRateCritic",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTabularMDAdvRateCritic:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
