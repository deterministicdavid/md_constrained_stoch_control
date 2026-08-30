from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.stats import norm
import torch
from sde_md.envs.sde_env import (
    drift_fn,
    diffusion_fn,
    running_cost_fn,
    terminal_cost_fn,
)

# for evaluation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sde_md.envs.sde_env import SDEDiffusionEnv

from sde_md.solvers.cont_act_fdm_solver import ContinuousActionsFDMValueEvaluator
from sde_md.iterators.dpp_based_iterations import DPPBasedIterations


class ContinuousActionMirrorDescent(DPPBasedIterations):
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
        action_radius=1.0,
        **kwargs,
    ):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.policy = policy
        self.ent_coef = ent_coef

        self.action_radius = action_radius
        self.cost_factor = getattr(self.env, "control_cost_factor", 0.1)
        # super().__init__(*args, **kwargs)

        # Override the simulation environemnt with the continuous actions one
        self.eval_freq = eval_freq
        self.vec_env = None
        if self.eval_freq > 0 and env_cfg is not None:
            print(
                f"Setting up {n_envs} parallel environments (Continuous) for MC Evaluation..."
            )
            self.vec_env = make_vec_env(
                SDEDiffusionEnv,
                n_envs=n_envs,
                env_kwargs=env_cfg,
                vec_env_cls=SubprocVecEnv,
            )

        # Override the FDM solver with the continuous actions one
        self.fdm_solver = ContinuousActionsFDMValueEvaluator(
            self.env,
            self.policy,
            grid_bins,
            gamma=gamma,
            ent_coef=self.ent_coef,
            action_radius=action_radius,
        )

    def _update_policy_params(
        self,
        Q_rate_x: np.ndarray,
        Q_rate_y: np.ndarray,
        current_slice_params: np.ndarray,
        n_x: int,
        **kwargs,
    ) -> np.ndarray:
        """Dummy to satisfy the ABC. Continuous iteration overrides iterate() entirely and updates happen in `_update_dual_space_params`."""
        return np.array([])

    def _update_dual_space_params(
        self, current_Z, dV_dx_f, dV_dx_b, dV_dy_f, dV_dy_b, md_lambda=0.5, **kwargs
    ):
        """
        Computes the Mirror Flow ODE update step for the dual variable Z.
        """
        # Map Z to physical actions u using the gradient of the conjugate function
        R2 = self.action_radius**2
        Z_norm_sq = np.sum(current_Z**2, axis=-1, keepdims=True)
        denom = 1.0 + np.sqrt(1.0 + R2 * Z_norm_sq)
        u = (R2 * current_Z) / denom

        # Select correct upwind gradient based on the direction of physical action u
        grad_V_x = np.where(u[:, 0] >= 0, dV_dx_f, dV_dx_b)
        grad_V_y = np.where(u[:, 1] >= 0, dV_dy_f, dV_dy_b)

        # Gradient of the Bregman Regularizer: nabla psi(u) = 2u / (R^2 - |u|^2)
        # tau_term = self.ent_coef * (2 * u) / (R2 - np.sum(u**2, axis=-1, keepdims=True) + 1e-12)
        tau_term = self.ent_coef * current_Z

        # Mirror Flow Update: dZ/ds = -(N^T nabla V + u + tau * nabla psi(u))
        # (Assuming N is the Identity matrix for the environment drift)
        dZ_x = -(grad_V_x + 2.0 * self.cost_factor * u[:, 0] + tau_term[:, 0])
        dZ_y = -(grad_V_y + 2.0 * self.cost_factor * u[:, 1] + tau_term[:, 1])

        new_Z_x = current_Z[:, 0] - md_lambda * dZ_x
        new_Z_y = current_Z[:, 1] - md_lambda * dZ_y

        return np.column_stack((new_Z_x, new_Z_y))

    def iterate(
        self, n_iterations=50, eval_seed: int = 42, eval_N_mc: int = 1024, **kwargs
    ):
        V = None
        V_prev = None
        n_x = len(self.fdm_solver.u_x_grid)

        for i in range(n_iterations):
            # 1. Policy Evaluation
            V = self.fdm_solver.solve()

            log_line = f"Iter: {i} out of {n_iterations}, "
            # --- 0. MC Evaluation Run ---
            if self.vec_env is not None and i % self.eval_freq == 0:
                mc_mean, mc_std = self._run_mc_eval(seed=eval_seed, N_mc=eval_N_mc)
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

            # 2. Grab Current Parameters (Z and V)
            with torch.no_grad():
                current_params = (
                    self.policy.tabular_params["Z_table"].cpu().numpy().copy()
                )
                new_values = (
                    self.policy.tabular_params["value_table"].cpu().numpy().copy()
                )

            new_params = current_params.copy()

            # Time Step Loop
            for k in range(self.fdm_solver.time_steps + 1):
                t = k * self.fdm_solver.dt
                V_k = V[k]

                dx, dy = self.fdm_solver.dx, self.fdm_solver.dy

                # Compute Spatial Differences
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

                # Fetch current Z
                current_Z = self.policy.get_action_Z_np(obs_tensor)

                new_Z = self._update_dual_space_params(
                    current_Z,
                    dV_dx_f.flatten(),
                    dV_dx_b.flatten(),
                    dV_dy_f.flatten(),
                    dV_dy_b.flatten(),
                    **kwargs,
                )

                new_params[policy_indices, 0] = new_Z[:, 0]
                new_params[policy_indices, 1] = new_Z[:, 1]
                new_values[policy_indices, 0] = V_k.flatten()

            # Write back to PyTorch
            with torch.no_grad():
                self.policy.tabular_params["Z_table"].copy_(
                    torch.tensor(
                        new_params, dtype=torch.float32, device=self.policy.device
                    )
                )
                self.policy.tabular_params["value_table"].copy_(
                    torch.tensor(
                        new_values, dtype=torch.float32, device=self.policy.device
                    )
                )


if __name__ == "__main__":
    print("Don't run directly, use run_dpp_based.py")
