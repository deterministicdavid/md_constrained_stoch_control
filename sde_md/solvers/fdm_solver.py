from abc import ABC, abstractmethod

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


@njit
def precompute_terminals(states_2d, terminal_cost_factor):
    """Fast C-loop for terminal conditions."""
    n = states_2d.shape[0]
    terminals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        terminals[i] = -terminal_cost_fn(states_2d[i], terminal_cost_factor)
    return terminals


@njit
def precompute_discrete_physics(states_2d, u_x_grid, u_y_grid, control_cost_factor):
    """Bypasses millions of Python loops by precomputing the (States x Actions) matrices in C."""
    n_states = states_2d.shape[0]
    p = len(u_x_grid) * len(u_y_grid)

    drift_x = np.zeros((n_states, p), dtype=np.float64)
    drift_y = np.zeros((n_states, p), dtype=np.float64)
    reward = np.zeros((n_states, p), dtype=np.float64)

    for idx_x in range(len(u_x_grid)):
        ux = u_x_grid[idx_x]
        for idx_y in range(len(u_y_grid)):
            uy = u_y_grid[idx_y]
            flat_a = idx_x * len(u_y_grid) + idx_y
            a_arr = np.array([ux, uy], dtype=np.float32)

            for i in range(n_states):
                d = drift_fn(states_2d[i], a_arr)
                drift_x[i, flat_a] = d[0]
                drift_y[i, flat_a] = d[1]
                reward[i, flat_a] = -running_cost_fn(
                    states_2d[i], a_arr, control_cost_factor
                )

    return drift_x, drift_y, reward


@njit
def build_upwind_diagonals(b_x, b_y, is_inside, grid_bins, dx, dy, sigma, gamma, dt):
    """The biggest bottleneck: Calculates the finite-difference diagonals in raw C."""
    n_states = grid_bins * grid_bins
    main_diag = np.zeros(n_states, dtype=np.float64)
    off_x_minus = np.zeros(n_states, dtype=np.float64)
    off_x_plus = np.zeros(n_states, dtype=np.float64)
    off_y_minus = np.zeros(n_states, dtype=np.float64)
    off_y_plus = np.zeros(n_states, dtype=np.float64)

    diff_x = (sigma**2) / (2 * dx**2)
    diff_y = (sigma**2) / (2 * dy**2)
    gamma_val = gamma if gamma < 1.0 else 0.0

    for k in range(n_states):
        if not is_inside[k]:
            main_diag[k] = 1.0
            continue

        i = k // grid_bins
        j = k % grid_bins

        cx = b_x[k]
        cy = b_y[k]

        A_kk = -gamma_val - 2 * diff_x - 2 * diff_y - abs(cx) / dx - abs(cy) / dy

        if i > 0:
            off_x_minus[k] = diff_x - min(cx, 0.0) / dx
        if i < grid_bins - 1:
            off_x_plus[k] = diff_x + max(cx, 0.0) / dx
        if j > 0:
            off_y_minus[k] = diff_y - min(cy, 0.0) / dy
        if j < grid_bins - 1:
            off_y_plus[k] = diff_y + max(cy, 0.0) / dy

        main_diag[k] = 1.0 - dt * A_kk
        off_x_minus[k] = -dt * off_x_minus[k]
        off_x_plus[k] = -dt * off_x_plus[k]
        off_y_minus[k] = -dt * off_y_minus[k]
        off_y_plus[k] = -dt * off_y_plus[k]

    return main_diag, off_x_minus, off_x_plus, off_y_minus, off_y_plus


class FDMValueEvaluator:
    def __init__(self, env, policy, grid_bins=40, gamma=1.0, ent_coef=1e-4):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.policy = policy
        self.grid_bins = grid_bins
        self.gamma = gamma
        self.tau = ent_coef

        self.R = self.env.radius
        self.sigma = self.env.sigma
        self.dt = self.env.dt
        self.T = self.env.T
        self.time_steps = int(np.ceil(self.T / self.dt))

        self.dx = (2 * self.R) / (self.grid_bins - 1)
        self.dy = (2 * self.R) / (self.grid_bins - 1)

        self.x_axis = np.linspace(-self.R, self.R, self.grid_bins)
        self.y_axis = np.linspace(-self.R, self.R, self.grid_bins)

        self.X, self.Y = np.meshgrid(self.x_axis, self.y_axis, indexing="ij")

        self.n_states = self.grid_bins**2
        self.flat_x = self.X.flatten()
        self.flat_y = self.Y.flatten()

        self.is_inside = (self.flat_x**2 + self.flat_y**2) < self.R**2

        self.is_multi_discrete = hasattr(self.env, "action_grids")
        if self.is_multi_discrete:
            self.u_x_grid = self.env.action_grids[0]
            self.u_y_grid = self.env.action_grids[1]
            self.p = len(self.u_x_grid) * len(self.u_y_grid)
        else:
            raise NotImplementedError("FDM expects a discrete action grid.")

        self._precompute_physics()

    def _precompute_physics(self):
        """Builds fixed matrices for drift, running reward, and terminal reward via Numba."""
        self.states_2d = np.column_stack((self.flat_x, self.flat_y)).astype(np.float32)

        self.terminal_rewards = precompute_terminals(
            self.states_2d, self.env.space_cost_factor_terminal
        )

        self.precomputed_drift_x, self.precomputed_drift_y, self.precomputed_reward = (
            precompute_discrete_physics(
                self.states_2d,
                self.u_x_grid,
                self.u_y_grid,
                self.env.control_cost_factor,
            )
        )

        self.obs_batch = np.zeros((self.n_states, 3), dtype=np.float32)
        self.obs_batch[:, 0] = self.flat_x
        self.obs_batch[:, 1] = self.flat_y

    def solve(self):
        # print(f"Starting FDM Ground Truth evaluation (Grid: {self.grid_bins}x{self.grid_bins}, Steps: {self.time_steps})")

        V_sol = np.zeros((self.time_steps + 1, self.n_states))

        # Apply exact terminal condition
        V_sol[-1, :] = self.terminal_rewards
        prior_prob = 1.0 / self.p

        for k in range(self.time_steps - 1, -1, -1):
            t = k * self.dt

            self.obs_batch[:, 2] = t
            obs_tensor = torch.as_tensor(self.obs_batch, device=self.policy.device)

            with torch.no_grad():
                action_probs, _ = self.policy.get_action_probs(obs_tensor)
                action_probs = action_probs.cpu().numpy().astype(np.float64)

                if self.is_multi_discrete:
                    n_x = len(self.u_x_grid)
                    prob_x = action_probs[:, :n_x]
                    prob_y = action_probs[:, n_x:]
                    joint_probs = np.einsum("bi,bj->bij", prob_x, prob_y).reshape(
                        self.n_states, self.p
                    )
                else:
                    joint_probs = action_probs

            # Vectorized Expected Physics
            b_x = np.sum(joint_probs * self.precomputed_drift_x, axis=1)
            b_y = np.sum(joint_probs * self.precomputed_drift_y, axis=1)

            # Avoid log(0) explicitly using safe masking for KL Divergence
            safe_probs = np.maximum(joint_probs, 1e-12)
            kl_matrix = joint_probs * np.log(safe_probs / prior_prob)
            R_pi = np.sum(
                joint_probs * self.precomputed_reward - self.tau * kl_matrix, axis=1
            )

            M = self._build_upwind_matrix(b_x, b_y)
            RHS = V_sol[k + 1, :] + self.dt * R_pi
            RHS[~self.is_inside] = self.terminal_rewards[~self.is_inside]

            # Iterative BiCGSTAB Solver (Uses future step V_sol[k+1] as initial guess)
            v_next, info = splinalg.bicgstab(M, RHS, x0=V_sol[k + 1, :], rtol=1e-10)

            if info == 0:
                V_sol[k, :] = v_next
            else:
                print("Falling back to direct solver. did M become ill-conditioned?")
                V_sol[k, :] = splinalg.spsolve(M, RHS)

        return V_sol.reshape(self.time_steps + 1, self.grid_bins, self.grid_bins)

    def _build_upwind_matrix(self, b_x, b_y):
        # ... [EXACTLY THE SAME AS PREVIOUS VERSION] ...
        main_diag = np.zeros(self.n_states)
        off_x_minus = np.zeros(self.n_states)
        off_x_plus = np.zeros(self.n_states)
        off_y_minus = np.zeros(self.n_states)
        off_y_plus = np.zeros(self.n_states)

        diff_x = (self.sigma**2) / (2 * self.dx**2)
        diff_y = (self.sigma**2) / (2 * self.dy**2)

        for i in range(self.grid_bins):
            for j in range(self.grid_bins):
                k = i * self.grid_bins + j
                if not self.is_inside[k]:
                    main_diag[k] = 1.0
                    continue

                cx = b_x[k]
                cy = b_y[k]

                A_kk = -(self.gamma if self.gamma < 1.0 else 0.0)
                A_kk -= 2 * diff_x + 2 * diff_y
                A_kk -= abs(cx) / self.dx + abs(cy) / self.dy

                # if i > 0:
                #     off_x_minus[k] = diff_x + max(cx, 0) / self.dx
                # if i < self.grid_bins - 1:
                #     off_x_plus[k] = diff_x - min(cx, 0) / self.dx
                # if j > 0:
                #     off_y_minus[k] = diff_y + max(cy, 0) / self.dy
                # if j < self.grid_bins - 1:
                #     off_y_plus[k] = diff_y - min(cy, 0) / self.dy

                if i > 0:
                    # Look left ONLY if drift is negative
                    off_x_minus[k] = diff_x - min(cx, 0) / self.dx
                if i < self.grid_bins - 1:
                    # Look right ONLY if drift is positive
                    off_x_plus[k] = diff_x + max(cx, 0) / self.dx
                if j > 0:
                    off_y_minus[k] = diff_y - min(cy, 0) / self.dy
                if j < self.grid_bins - 1:
                    off_y_plus[k] = diff_y + max(cy, 0) / self.dy

                main_diag[k] = 1.0 - self.dt * A_kk
                off_x_minus[k] = -self.dt * off_x_minus[k]
                off_x_plus[k] = -self.dt * off_x_plus[k]
                off_y_minus[k] = -self.dt * off_y_minus[k]
                off_y_plus[k] = -self.dt * off_y_plus[k]

        diagonals = [
            main_diag,
            off_x_minus[self.grid_bins :],
            off_x_plus[: -self.grid_bins],
            off_y_minus[1:],
            off_y_plus[:-1],
        ]
        offsets = [0, -self.grid_bins, self.grid_bins, -1, 1]

        M = sp.diags(
            diagonals, offsets, shape=(self.n_states, self.n_states), format="csr"
        )
        return M


if __name__ == "__main__":
    print("Don't run directly, use run_dpp_based.py")
