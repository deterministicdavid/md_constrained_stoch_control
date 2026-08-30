import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

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

from stable_baselines3.common.buffers import ReplayBuffer
from typing import NamedTuple

SelfTabularMDAdvRateCriticTDL = TypeVar(
    "SelfTabularMDAdvRateCriticTDL", bound="TabularMDAdvRateCriticTDL"
)


class SequenceReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class SequenceReplayBuffer(ReplayBuffer):
    def __init__(self, *args, seq_len: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len

    def sample(self, batch_size: int, env=None) -> SequenceReplayBufferSamples:
        # 1. Determine valid start indices avoiding the write-pointer
        if self.full:
            valid_inds = np.arange(self.buffer_size)
            invalid_inds = np.array(
                [(self.pos - i) % self.buffer_size for i in range(1, self.seq_len + 1)]
            )
            valid_inds = np.setdiff1d(valid_inds, invalid_inds)
        else:
            valid_inds = np.arange(max(0, self.pos - self.seq_len))
            if len(valid_inds) == 0:
                valid_inds = np.array([0])

        # 2. Sample random starts and environments
        batch_inds = np.random.choice(valid_inds, size=batch_size)
        env_inds = np.random.randint(0, self.n_envs, size=batch_size)

        # --- 3. VECTORIZED INDEXING (No Python Loops!) ---
        # Create a 2D matrix of indices: shape [batch_size, seq_len]
        seq_offsets = np.arange(self.seq_len)
        idx_2d = (batch_inds[:, None] + seq_offsets[None, :]) % self.buffer_size

        # Expand env_inds to match the 2D shape
        env_2d = np.broadcast_to(env_inds[:, None], (batch_size, self.seq_len))

        # 4. Instantly slice the underlying C-arrays
        obs_seq = self.observations[idx_2d, env_2d]
        next_obs_seq = self.next_observations[idx_2d, env_2d]
        actions_seq = self.actions[idx_2d, env_2d]
        rewards_seq = self.rewards[idx_2d, env_2d]
        dones_seq = self.dones[idx_2d, env_2d]

        def to_tensor(arr):
            return th.as_tensor(arr, device=self.device)

        return SequenceReplayBufferSamples(
            observations=to_tensor(obs_seq),
            actions=to_tensor(actions_seq),
            next_observations=to_tensor(next_obs_seq),
            dones=to_tensor(dones_seq).float(),
            rewards=to_tensor(rewards_seq),
        )


class TabularMDAdvRateCriticTDL(OffPolicyAlgorithm):

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
        seq_len: int = 4,  # for TD learning, how many steps
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
        self.seq_len = seq_len
        super().__init__(
            policy,
            env,
            replay_buffer_class=SequenceReplayBuffer,
            replay_buffer_kwargs={"seq_len": self.seq_len},
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
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            B, L, obs_dim = replay_data.observations.shape

            # --- 1. Evaluate Sequence Integrals (Fully Vectorized) ---
            with th.no_grad():
                # Flatten sequences into a single huge batch for blazing fast network passes
                obs_flat = replay_data.observations.view(B * L, obs_dim)
                actions_flat = replay_data.actions.view(B * L, -1).long()

                # Quantize all states at once
                state_indices_flat = self.policy._quantize(obs_flat)
                state_indices = state_indices_flat.view(B, L)

                col_indices_flat = actions_flat + logit_offsets_tensor
                col_indices = col_indices_flat.view(B, L, -1)

                # Fetch Q-values for all states in all sequences simultaneously
                q_flat = self.policy.critic(state_indices_flat)
                q_taken_flat = q_flat.gather(1, col_indices_flat)
                q_taken_joint_flat = q_taken_flat.sum(dim=-1, keepdim=True)
                q_taken_joint = q_taken_joint_flat.view(
                    B, L
                )  # Reshape back to sequence

                # Compute discount masks efficiently using cumulative products
                not_dones = 1.0 - replay_data.dones  # [B, L]
                # Shift right by 1: first step discount is always gamma^0 = 1
                shifted_not_dones = th.cat(
                    [th.ones((B, 1), device=self.device), not_dones[:, :-1]], dim=1
                )
                cum_not_dones = th.cumprod(shifted_not_dones, dim=1)

                # discounts shape: [B, L]
                discounts = (
                    self.gamma ** th.arange(L, device=self.device).float().unsqueeze(0)
                ) * cum_not_dones

                # Vectorized Integral: sum over the L dimension
                step_integrals = discounts * (replay_data.rewards - q_taken_joint) * dt
                cumulative_integral = step_integrals.sum(dim=1, keepdim=True)  # [B, 1]

                # Terminal V (Discount for the state exactly AFTER the sequence ends)
                terminal_discount = (
                    discounts[:, -1:] * self.gamma * not_dones[:, -1:]
                )  # [B, 1]
                last_next_obs = replay_data.next_observations[:, -1, :]
                last_next_state_indices = self.policy._quantize(last_next_obs)
                next_v_term = self.policy.v_critic_target(last_next_state_indices)

                current_v = self.policy.v_critic(state_indices[:, 0])  # [B, 1]

                # Final Equation 4.3 Loss
                martingale_td_error = (
                    current_v - cumulative_integral - terminal_discount * next_v_term
                )  # [B, 1]

            lr = self.learning_rate(self._current_progress_remaining)

            # --- 2. Update V-Function (Only at start of sequence: x_0) ---
            with th.no_grad():
                q_critic_losses.append((martingale_td_error**2).mean().item())

                state_indices_0 = state_indices[:, 0].long()
                state_counts_0 = th.bincount(state_indices_0)
                batch_state_counts_0 = state_counts_0[state_indices_0].unsqueeze(1)

                v_updates = (lr * martingale_td_error) / batch_state_counts_0
                self.policy.v_critic.v_table.view(-1).index_add_(
                    0, state_indices_0, -v_updates.view(-1)
                )

            # --- 3. Update Q and Actor (Vectorized across all valid sequence steps) ---
            with th.no_grad():
                # Filter out padded sequence steps (where discount is 0)
                valid_mask = discounts > 0  # [B, L]

                active_states = state_indices[valid_mask].long()  # [N_active]
                active_cols = col_indices[valid_mask]  # [N_active, n_dims]
                active_disc = discounts[valid_mask]  # [N_active]

                # Expand the batch-level TD error to all L steps, then filter
                active_err = martingale_td_error.expand(B, L)[valid_mask]  # [N_active]

                # --- 3a. Q-Table Update ---
                flat_sa_indices = (
                    active_states.unsqueeze(1) * total_logit_cols + active_cols
                ).reshape(-1)
                sa_counts = th.bincount(flat_sa_indices)
                batch_sa_counts = sa_counts[flat_sa_indices]

                expanded_err = (
                    active_err.unsqueeze(1).expand_as(active_cols).reshape(-1)
                )
                expanded_disc = (
                    active_disc.unsqueeze(1).expand_as(active_cols).reshape(-1)
                )

                # Apply gradients specifically factored by the discount at step `l`
                q_updates = (lr * expanded_err * expanded_disc * dt) / batch_sa_counts
                self.policy.critic.q_table.view(-1).index_add_(
                    0, flat_sa_indices, -q_updates
                )

                # --- 3b. Actor Update ---
                # We fetch log probs for only the active steps in a single batched pass
                active_obs = replay_data.observations[valid_mask]
                _, active_log_probs = self.policy.get_action_probs(active_obs)

                fresh_q = self.policy.critic.q_table[active_states]
                current_v_sync = self.policy.v_critic(active_states)

                first_variation = (
                    fresh_q - current_v_sync - self.ent_coef * active_log_probs
                )

                state_counts = th.bincount(active_states)
                batch_state_counts = state_counts[active_states].unsqueeze(1)

                actor_updates = (first_variation * md_lambda_val) / batch_state_counts
                self.policy.actor.action_logits.index_add_(
                    0, active_states, actor_updates
                )

                # Shift logits to log-probabilities once at the end of the gradient step
                nvec_arg = (
                    self.action_space.nvec
                    if isinstance(self.action_space, spaces.MultiDiscrete)
                    else None
                )
                self.policy.actor.normalize_policy_(nvec=nvec_arg)

            # --- 4. Update Target Value Network (Polyak) ---
            polyak_update(
                self.policy.v_critic.parameters(),
                self.policy.v_critic_target.parameters(),
                self.tau,
            )
            v_v_tgt_gap = (
                (
                    (
                        self.policy.v_critic(state_indices[:, 0])
                        - self.policy.v_critic_target(state_indices[:, 0])
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
        self: SelfTabularMDAdvRateCriticTDL,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TabularMDAdvRateCritic",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTabularMDAdvRateCriticTDL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
