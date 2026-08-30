import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.stats import norm
import torch

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sde_md.envs.sde_env import SDEDiffusionEnv

from sde_md.solvers.cont_act_fdm_solver import ContinuousActionsFDMValueEvaluator
from sde_md.iterators.dpp_based_iterations import DPPBasedIterations


class ContinuousActionPolicyIteration(DPPBasedIterations):
    """
    Solves the continuous-action MDP using Exact Policy Iteration.
    Directly maximizes the Hamiltonian a |-> H(t, x, nabla V, a) over the 
    action ball, bypassing the need for a learning rate.
    """

    def __init__(
        self,
        env,
        policy,
        grid_bins=40,
        gamma=1.0,
        ent_coef=0,
        n_envs=1,
        eval_freq=0,
        env_cfg=None,
        action_radius=1.0,
        policy_table_key="action_table",  # Key for the primal action table in your policy
        **kwargs,
    ):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.policy = policy
        self.ent_coef = ent_coef
        self.action_radius = action_radius
        self.cost_factor = getattr(self.env, "control_cost_factor", 0.1)
        self.policy_table_key = policy_table_key

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

        self.fdm_solver = ContinuousActionsFDMValueEvaluator(
            self.env,
            self.policy,
            grid_bins,
            gamma=gamma,
            ent_coef=self.ent_coef,
            action_radius=action_radius,
        )

    def _update_policy_params(self, *args, **kwargs) -> np.ndarray:
        """Dummy to satisfy the ABC."""
        return np.array([])

    def _update_primal_space_params(
        self, current_a, dV_dx_f, dV_dx_b, dV_dy_f, dV_dy_b, **kwargs
    ):
        """
        Computes the exact Policy Iteration update step.
        """
        # 1. Select upwind gradient based on the direction of the CURRENT action a.
        # (This is standard lagged-upwinding for stable HJB Policy Iteration)
        grad_V_x = np.where(current_a[:, 0] >= 0, dV_dx_f, dV_dx_b)
        grad_V_y = np.where(current_a[:, 1] >= 0, dV_dy_f, dV_dy_b)

        # 2. Exact unconstrained maximizer of the Hamiltonian
        # H(x,a) = a * nabla V - r * |a|^2  =>  a* = nabla V / (2r)
        new_a_x = grad_V_x / (2.0 * self.cost_factor)
        new_a_y = grad_V_y / (2.0 * self.cost_factor)
        new_a = np.column_stack((new_a_x, new_a_y))

        # 3. Explicit Projection onto the L2 Action Ball
        a_norm = np.linalg.norm(new_a, axis=1, keepdims=True)
        # Scale down only if the norm exceeds the action_radius
        scale = np.minimum(1.0, self.action_radius / np.maximum(a_norm, 1e-12))
        projected_a = new_a * scale

        return projected_a

    def iterate(
        self, n_iterations=50, eval_seed: int = 42, eval_N_mc: int = 1024, **kwargs
    ):
        V = None
        V_prev = None

        table_key = (
            self.policy_table_key
            if self.policy_table_key in self.policy.tabular_params
            else "Z_table"
        )

        for i in range(n_iterations):
            # 1. Policy Evaluation
            V = self.fdm_solver.solve()

            log_line = f"Iter: {i} out of {n_iterations}, "

            # --- MC Evaluation Run ---
            if self.vec_env is not None and i % self.eval_freq == 0:
                # Assuming run_mc_eval is imported or available on self depending on your class hierarchy
                mc_mean, mc_std = self._run_mc_eval(seed=eval_seed, N_mc=eval_N_mc)
                conf_level = 0.95
                z_score = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)
                margin_of_error = z_score * (mc_std / np.sqrt(eval_N_mc))
                log_line += f" MC Eval: Average Return = {mc_mean:.2f} +/- {margin_of_error:.2f} ({conf_level*100:.0f}% CI)"

            if V_prev is not None:
                sup_norm = np.max(np.abs(V - V_prev))
                l2_norm = np.sqrt(np.mean((V - V_prev) ** 2))
                log_line += f" Value Change -> Sup Norm: {sup_norm:.2e} | L2 Norm: {l2_norm:.2e}"

            V_prev = V.copy()
            print(log_line, end="\n", flush=True)

            # 2. Grab Current Parameters
            with torch.no_grad():
                current_params = (
                    self.policy.tabular_params[table_key].cpu().numpy().copy()
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

                # Compute Spatial Differences (Upwind FDM)
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

                if hasattr(self.policy, "get_actions_np"):
                    current_a = self.policy.get_actions_np(obs_tensor)
                else:
                    current_a = current_params[policy_indices]

                # Policy Iteration Exact Update
                new_a = self._update_primal_space_params(
                    current_a,
                    dV_dx_f.flatten(),
                    dV_dx_b.flatten(),
                    dV_dy_f.flatten(),
                    dV_dy_b.flatten(),
                    **kwargs,  # Consumes any lingering swept parameters gracefully
                )

                new_params[policy_indices, 0] = new_a[:, 0]
                new_params[policy_indices, 1] = new_a[:, 1]
                new_values[policy_indices, 0] = V_k.flatten()

            # 3. Write back to PyTorch
            with torch.no_grad():
                self.policy.tabular_params[table_key].copy_(
                    torch.tensor(
                        new_params, dtype=torch.float32, device=self.policy.device
                    )
                )
                self.policy.tabular_params["value_table"].copy_(
                    torch.tensor(
                        new_values, dtype=torch.float32, device=self.policy.device
                    )
                )