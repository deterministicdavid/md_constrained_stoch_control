import numpy as np
from numba import njit
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

from sde_md.solvers.fdm_solver import FDMValueEvaluator


@njit
def compute_physics_batch(states, actions, control_cost_factor):
    n = states.shape[0]
    b_x = np.empty(n, dtype=np.float64)
    b_y = np.empty(n, dtype=np.float64)
    base_costs = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Calls the already-njitted functions directly in C space
        d = drift_fn(states[i], actions[i])
        b_x[i] = d[0]
        b_y[i] = d[1]
        base_costs[i] = running_cost_fn(states[i], actions[i], control_cost_factor)

    return b_x, b_y, base_costs


class ContinuousActionsFDMValueEvaluator(FDMValueEvaluator):
    def __init__(
        self, env, policy, grid_bins=40, gamma=1.0, ent_coef=1e-4, action_radius=1.0
    ):
        self.action_radius = action_radius
        # Bypasses the discrete grid requirement in the parent class
        env.action_grids = [np.array([0]), np.array([0])]
        super().__init__(env, policy, grid_bins, gamma, ent_coef)
        del env.action_grids

    def _precompute_physics(self):
        # We only precompute terminal rewards. Drift and running costs
        # depend on the continuous action u, so they are computed on the fly.
        self.terminal_rewards = np.zeros(self.n_states)
        self.states_2d = np.column_stack((self.flat_x, self.flat_y)).astype(np.float32)

        for i in range(self.n_states):
            self.terminal_rewards[i] = -terminal_cost_fn(
                self.states_2d[i], self.env.space_cost_factor_terminal
            )

        self.obs_batch = np.zeros((self.n_states, 3), dtype=np.float32)
        self.obs_batch[:, 0] = self.flat_x
        self.obs_batch[:, 1] = self.flat_y

    def solve(self):
        V_sol = np.zeros((self.time_steps + 1, self.n_states))
        V_sol[-1, :] = self.terminal_rewards

        for k in range(self.time_steps - 1, -1, -1):
            t = k * self.dt
            self.obs_batch[:, 2] = t
            obs_tensor = torch.as_tensor(self.obs_batch, device=self.policy.device)

            with torch.no_grad():
                actions, _, _ = self.policy(obs_tensor)
                actions_np = actions.cpu().numpy().astype(np.float64)

            # Batched Physics (Massive Speedup)
            b_x, b_y, base_costs = compute_physics_batch(
                self.states_2d, actions_np, self.env.control_cost_factor
            )

            # Vectorized Bregman Divergence (No Python loop)
            if self.tau > 0:
                a_sq = np.sum(actions_np**2, axis=1)
                safe_a_sq = np.clip(a_sq, 0.0, self.action_radius**2 - 1e-12)
                d_psi = np.log(
                    self.action_radius**2 / (self.action_radius**2 - safe_a_sq)
                )
                base_costs += self.tau * d_psi

            R_pi = -base_costs

            M = self._build_upwind_matrix(b_x, b_y)
            RHS = V_sol[k + 1, :] + self.dt * R_pi
            RHS[~self.is_inside] = self.terminal_rewards[~self.is_inside]

            v_next, info = splinalg.bicgstab(M, RHS, x0=V_sol[k + 1, :], rtol=1e-8)

            if info == 0:
                V_sol[k, :] = v_next
            else:
                print("Falling back to direct solver. did M become ill-conditioned?")
                V_sol[k, :] = splinalg.spsolve(M, RHS)

        return V_sol.reshape(self.time_steps + 1, self.grid_bins, self.grid_bins)


if __name__ == "__main__":
    print("Don't run directly, use run_dpp_based.py")
