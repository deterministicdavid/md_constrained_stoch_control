from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.stats import norm
import torch

from sde_md.envs.sde_env import (
    SDEDiffusionEnvDiscAct,
    drift_fn,
    diffusion_fn,
    running_cost_fn,
    terminal_cost_fn,
)

# for evaluation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


from sde_md.solvers.fdm_solver import FDMValueEvaluator



def run_mc_eval(
    vec_env,
    policy,
    t0: Union[float, None],
    x0: Union[np.array, None],
    seed: Union[int, None],
    N_mc: int = 1024,
):
    """Runs exact MC evaluation with reproducible noise/initial states."""
    # Force the exact same seed right before evaluation to guarantee
    # identical starting states and SDE noise trajectories across iterations!
    if seed is not None:
        vec_env.seed(seed)

    obs = None
    if (t0 is not None) and (x0 is not None):
        # obs = vec_env.reset(options={"x0": x0, "t0": t0})
        reset_results = vec_env.env_method("reset", options={"x0": x0, "t0": t0})
        obs = np.stack([res[0] for res in reset_results])
    else:
        obs = vec_env.reset()

    episode_rewards = []
    current_rewards = np.zeros(vec_env.num_envs)
    episodes_completed = 0

    while episodes_completed < N_mc:
        # Predict actions using current policy
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
            # deterministic=False samples from the current softmax logits
            # actions, _, _ = policy(obs_tensor, deterministic=False)
            actions = policy._predict(obs_tensor, deterministic=False)
            actions = actions.cpu().numpy()

        obs, rewards, dones, infos = vec_env.step(actions)
        current_rewards += rewards

        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0.0
                episodes_completed += 1

                if episodes_completed >= N_mc:
                    break

                if (x0 is not None) and (t0 is not None):
                    reset_res = vec_env.env_method(
                        "reset", options={"x0": x0, "t0": t0}, indices=[i]
                    )[
                        0
                    ]  # env_method returns a list of results, grab the 0th

                    obs[i] = reset_res[0]

    return np.mean(episode_rewards[:N_mc]), np.std(episode_rewards[:N_mc])


class DPPBasedIterations(ABC):
    """
    Abstract base class for Dynamic Programming based policy iteration methods.
    Handles PDE solving, exact upwind gradient computation, and physics mappings.
    """

    def __init__(
        self,
        env,
        policy,
        grid_bins=40,
        gamma=1.0,
        ent_coef=1e-4,
        n_envs=1,
        eval_freq=0,
        env_cfg=None,
    ):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.policy = policy
        self.ent_coef = ent_coef

        self.fdm_solver = FDMValueEvaluator(
            self.env, policy, grid_bins, gamma, ent_coef
        )

        self._precompute_separable_physics()

        # --- MC Eval Setup ---
        self.vec_env = None
        self.eval_freq = eval_freq
        if self.eval_freq > 0 and env_cfg is not None:
            print(f"Setting up {n_envs} parallel environments for MC Evaluation...")
            self.vec_env = make_vec_env(
                SDEDiffusionEnvDiscAct,
                n_envs=n_envs,
                env_kwargs=env_cfg,
                vec_env_cls=SubprocVecEnv,
            )

    def _precompute_separable_physics(self):
        n_states = self.fdm_solver.n_states
        n_x = len(self.fdm_solver.u_x_grid)
        n_y = len(self.fdm_solver.u_y_grid)

        states_2d = np.column_stack(
            (self.fdm_solver.flat_x, self.fdm_solver.flat_y)
        ).astype(np.float32)

        self.drift_x_isolated = np.zeros((n_states, n_x))
        self.cost_x_isolated = np.zeros((n_states, n_x))

        self.drift_y_isolated = np.zeros((n_states, n_y))
        self.cost_y_isolated = np.zeros((n_states, n_y))

        for idx_x, ux in enumerate(self.fdm_solver.u_x_grid):
            for i in range(n_states):
                a_arr = np.array([ux, 0.0], dtype=np.float32)
                d = drift_fn(states_2d[i], a_arr)
                self.drift_x_isolated[i, idx_x] = d[0]
                self.cost_x_isolated[i, idx_x] = running_cost_fn(
                    states_2d[i], a_arr, self.env.control_cost_factor
                )

        for idx_y, uy in enumerate(self.fdm_solver.u_y_grid):
            for i in range(n_states):
                a_arr = np.array([0.0, uy], dtype=np.float32)
                d = drift_fn(states_2d[i], a_arr)
                self.drift_y_isolated[i, idx_y] = d[1]
                self.cost_y_isolated[i, idx_y] = running_cost_fn(
                    states_2d[i], a_arr, self.env.control_cost_factor
                )

    @abstractmethod
    def _update_policy_params(
        self,
        Q_rate_x: np.ndarray,
        Q_rate_y: np.ndarray,
        current_slice_params: np.ndarray,
        n_x: int,
        **kwargs,
    ) -> np.ndarray:
        pass

    def iterate(
        self, n_iterations=50, eval_seed: int = 42, eval_N_mc: int = 1024, **kwargs
    ):
        V = None
        V_prev = None
        n_x = len(self.fdm_solver.u_x_grid)

        for i in range(n_iterations):
            # 1. Policy Evaluation
            V = self.fdm_solver.solve()

            log_line = f"Iter: {i+1} out of {n_iterations}, "
            # --- 0. MC Evaluation Run ---
            if self.vec_env is not None and i % self.eval_freq == 0:
                mc_mean, mc_std = run_mc_eval(
                    self.vec_env, self.policy, seed=eval_seed, N_mc=eval_N_mc
                )
                conf_level = 0.95
                z_score = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)
                margin_of_error = z_score * (mc_std / np.sqrt(eval_N_mc))
                log_line += f" MC Eval: Average Return = {mc_mean:.2f} +/- {margin_of_error:.2f} ({conf_level*100:.0f}% CI)"

            if V_prev is not None:
                sup_norm = np.max(np.abs(V - V_prev))
                l2_norm = np.sqrt(np.mean((V - V_prev) ** 2))
                log_line += f" Value Function Change -> Sup Norm: {sup_norm:.2e} | L2 Norm: {l2_norm:.2e}"
                # print(f"Iter: {i}, Value Function Change -> Sup Norm: {sup_norm:.6f} | L2 Norm: {l2_norm:.6f}")

            V_prev = V.copy()
            print(log_line, end="\n", flush=True)

            # 2. Grab Current Parameters
            with torch.no_grad():
                if hasattr(self.policy, "action_logits"):
                    current_params = self.policy.action_logits.cpu().numpy().copy()
                    new_values = self.policy.value_table.cpu().numpy().copy()
                elif hasattr(self.policy, "tabular_params"):
                    is_prob_based = "action_probs" in self.policy.tabular_params
                    param_key = "action_probs" if is_prob_based else "action_logits"
                    current_params = (
                        self.policy.tabular_params[param_key].cpu().numpy().copy()
                    )
                    new_values = (
                        self.policy.tabular_params["value_table"].cpu().numpy().copy()
                    )
                else:
                    raise ValueError("Could not find tabular parameters.")

            new_params = current_params.copy()

            # 3. Time Step Loop for Upwind PDE logic
            for k in range(self.fdm_solver.time_steps + 1):
                t = k * self.fdm_solver.dt
                V_k = V[k]

                dx, dy = self.fdm_solver.dx, self.fdm_solver.dy

                dV_dx_f = np.zeros_like(V_k)
                dV_dx_f[:-1, :] = (V_k[1:, :] - V_k[:-1, :]) / dx
                dV_dx_f[-1, :] = dV_dx_f[-2, :]

                dV_dy_f = np.zeros_like(V_k)
                dV_dy_f[:, :-1] = (V_k[:, 1:] - V_k[:, :-1]) / dy
                dV_dy_f[:, -1] = dV_dy_f[:, -2]

                dV_dx_b = np.zeros_like(V_k)
                dV_dx_b[1:, :] = (V_k[1:, :] - V_k[:-1, :]) / dx
                dV_dx_b[0, :] = dV_dx_b[1, :]

                dV_dy_b = np.zeros_like(V_k)
                dV_dy_b[:, 1:] = (V_k[:, 1:] - V_k[:, :-1]) / dy
                dV_dy_b[:, 0] = dV_dy_b[:, 1]

                dV_dx_f_flat, dV_dx_b_flat = dV_dx_f.flatten(), dV_dx_b.flatten()
                dV_dy_f_flat, dV_dy_b_flat = dV_dy_f.flatten(), dV_dy_b.flatten()

                obs_batch = np.stack(
                    [
                        self.fdm_solver.flat_x,
                        self.fdm_solver.flat_y,
                        np.full_like(self.fdm_solver.flat_x, t),
                    ],
                    axis=1,
                )
                obs_tensor = torch.as_tensor(
                    obs_batch, dtype=torch.float32, device=self.policy.device
                )
                policy_indices = self.policy._quantize(obs_tensor).cpu().numpy()

                new_values[policy_indices, 0] = V_k.flatten()

                Q_rate_x = np.zeros(
                    (self.fdm_solver.n_states, len(self.fdm_solver.u_x_grid))
                )
                for ax_idx, ux in enumerate(self.fdm_solver.u_x_grid):
                    grad_x = dV_dx_f_flat if ux >= 0 else dV_dx_b_flat
                    Q_rate_x[:, ax_idx] = -grad_x * ux + self.cost_x_isolated[:, ax_idx]

                Q_rate_y = np.zeros(
                    (self.fdm_solver.n_states, len(self.fdm_solver.u_y_grid))
                )
                for ay_idx, uy in enumerate(self.fdm_solver.u_y_grid):
                    grad_y = dV_dy_f_flat if uy >= 0 else dV_dy_b_flat
                    Q_rate_y[:, ay_idx] = -grad_y * uy + self.cost_y_isolated[:, ay_idx]

                # --- DELEGATE LOGIT UPDATE TO SUBCLASS ---
                slice_params = current_params[policy_indices]
                joint_params_new = self._update_policy_params(
                    Q_rate_x,
                    Q_rate_y,
                    slice_params,
                    n_x,
                    V_k_flat=V_k.flatten(),
                    **kwargs,
                )
                new_params[policy_indices] = joint_params_new

            # 4. Write back to PyTorch, keeping each parameter's own dtype
            # (float32 or float64), so no precision is lost on the way back.
            with torch.no_grad():
                if hasattr(self.policy, "action_logits"):
                    target_params = self.policy.action_logits
                    target_values = self.policy.value_table
                elif hasattr(self.policy, "tabular_params"):
                    is_prob_based = "action_probs" in self.policy.tabular_params
                    param_key = "action_probs" if is_prob_based else "action_logits"
                    target_params = self.policy.tabular_params[param_key]
                    target_values = self.policy.tabular_params["value_table"]
                else:
                    raise ValueError("Could not find tabular parameters.")
                target_params.copy_(
                    torch.tensor(
                        new_params, dtype=target_params.dtype, device=self.policy.device
                    )
                )
                target_values.copy_(
                    torch.tensor(
                        new_values, dtype=target_values.dtype, device=self.policy.device
                    )
                )

            nvec_arg = getattr(self.policy.action_space, "nvec", None)
            self.policy.normalize_policy_(nvec=nvec_arg)

        if self.vec_env is not None:
            mc_mean, mc_std = run_mc_eval(
                self.vec_env, self.policy, seed=eval_seed, N_mc=eval_N_mc
            )
            conf_level = 0.95
            z_score = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)
            margin_of_error = z_score * (mc_std / np.sqrt(eval_N_mc))
            print(
                f"Final MC Eval: Average Return = {mc_mean:.2f} +/- {margin_of_error:.2f} ({conf_level*100:.0f}% CI)"
            )

        return V


if __name__ == "__main__":
    print("Don't run directly, use run_dpp_based.py")
