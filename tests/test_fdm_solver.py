import pytest
import numpy as np
import torch
import time
from sde_md.envs.sde_env import SDEDiffusionEnvDiscAct
from sde_md.policies.quantizedtabularpolicy import QuantizedTabularPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from sde_md.solvers.fdm_solver import FDMValueEvaluator
from sde_md.iterators.dpp_based_iterations import run_mc_eval


def init_heuristic_tabular_policy(policy, env):
    """Directly writes extreme logits into the QuantizedTabularPolicy to force it to play the deterministic heuristic."""
    fdm = FDMValueEvaluator(env, policy, grid_bins=policy.grid_bins)
    u_x_grid = np.array(fdm.u_x_grid)
    u_y_grid = np.array(fdm.u_y_grid)
    max_x = np.max(np.abs(u_x_grid))
    max_y = np.max(np.abs(u_y_grid))
    n_x, n_y = len(u_x_grid), len(u_y_grid)

    new_logits = np.zeros_like(
        policy.tabular_params["action_logits"].detach().cpu().numpy()
    )

    for k in range(fdm.time_steps + 1):
        t = k * fdm.dt
        r = np.sqrt(fdm.flat_x**2 + fdm.flat_y**2)
        r_safe = np.maximum(r, 1e-8)

        scale = (np.minimum(r, fdm.R) / fdm.R) * (t / fdm.T)
        scale = np.clip(scale, 0.0, 1.0)

        dir_x = -fdm.flat_x / r_safe
        dir_y = -fdm.flat_y / r_safe
        dir_x[r == 0] = 0.0
        dir_y[r == 0] = 0.0

        ideal_x = dir_x * scale * max_x
        ideal_y = dir_y * scale * max_y

        dist_x = np.abs(ideal_x[:, None] - u_x_grid[None, :])
        idx_x = np.argmin(dist_x, axis=1)

        dist_y = np.abs(ideal_y[:, None] - u_y_grid[None, :])
        idx_y = np.argmin(dist_y, axis=1)

        obs_batch = np.stack(
            [fdm.flat_x, fdm.flat_y, np.full_like(fdm.flat_x, t)], axis=1
        )
        obs_tensor = torch.as_tensor(
            obs_batch, dtype=torch.float32, device=policy.device
        )
        policy_indices = policy._quantize(obs_tensor).cpu().numpy()

        time_slice_logits = np.full((fdm.n_states, n_x + n_y), -100.0, dtype=np.float32)
        time_slice_logits[np.arange(fdm.n_states), idx_x] = 100.0
        time_slice_logits[np.arange(fdm.n_states), n_x + idx_y] = 100.0
        new_logits[policy_indices] = time_slice_logits

    with torch.no_grad():
        policy.tabular_params["action_logits"].copy_(
            torch.tensor(new_logits, device=policy.device)
        )
    nvec_arg = getattr(policy.action_space, "nvec", None)
    policy.normalize_policy_(nvec=nvec_arg)


def get_fdm_value(fdm_solver, V_sol, x, y, t):
    """Maps a physical (x, y, t) coordinate to the closest FDM grid index."""
    t_safe = min(max(t, 0.0), fdm_solver.T)
    t_idx = int(np.round(t_safe / fdm_solver.dt))
    x_idx = np.argmin(np.abs(fdm_solver.x_axis - x))
    y_idx = np.argmin(np.abs(fdm_solver.y_axis - y))
    return V_sol[t_idx, x_idx, y_idx]


# --- SHARED SETUP LOGIC ---


def setup_evaluation_environment(verbose=False):
    """Creates and wires up the solver, policy, and environments. Used by both Pytest and the debug runner."""
    if verbose:
        print("[1/3] Initializing Environment and Policy...")

    env_kwargs = {
        "radius": 5.0,
        "sigma": 0.05,
        "T": 1.0,
        "dt": 0.01,
        "action_min": -15.0,
        "action_max": 15.0,
        "n_actions_per_dim": 5,
        "state_dim": 2,
        "action_dim": 2,
        "control_cost_factor": 0.1,
    }
    env = SDEDiffusionEnvDiscAct(**env_kwargs)
    grid_bins = 161

    policy = QuantizedTabularPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.0,
        grid_bins=grid_bins,
        radius=env_kwargs["radius"],
        T=env_kwargs["T"],
        dt=env_kwargs["dt"],
        action_logits_to_optimizer=False,
    )
    policy.eval()
    init_heuristic_tabular_policy(policy, env)

    if verbose:
        print(f"[2/3] Solving FDM on {grid_bins}x{grid_bins} grid...")
        start_time = time.time()

    fdm_solver = FDMValueEvaluator(
        env=env, policy=policy, grid_bins=grid_bins, ent_coef=1e-6
    )
    V_sol = fdm_solver.solve()

    if verbose:
        print(f"      -> Done in {time.time() - start_time:.2f} seconds.\n")
        print(f"[3/3] Launching parallel environments for MC rollouts...")

    n_envs = 16
    vec_env = make_vec_env(
        SDEDiffusionEnvDiscAct,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    return fdm_solver, V_sol, vec_env, policy, env


# --- PYTEST CONFIGURATION & TESTS ---


@pytest.fixture(scope="module")
def shared_eval_setup():
    """Sets up the objects ONCE for the entire test module using the shared builder."""
    fdm_solver, V_sol, vec_env, policy, env = setup_evaluation_environment(
        verbose=False
    )
    yield fdm_solver, V_sol, vec_env, policy, env
    vec_env.close()


# Format: (x, y, t, absolute_tolerance, relative_tolerance)
TEST_POINTS = [
    # Origin
    (1, 0.2, 0.1, 0.0, 0.05, 0.05),
    (2, -0.2, 0.05, 0.5, 0.05, 0.50),
    (3, 0.15, 0.05, 0.9, 0.03, 0.40),
    # Mid-range points
    (4, 3.0, 3.0, 0.0, 0.20, 0.02),
    (5, 3.0, -3.0, 0.0, 0.20, 0.03),
    (6, -3.0, 3.0, 0.0, 0.20, 0.03),
    (7, 3.0, 3.0, 0.5, 0.20, 0.02),
    (8, 3.0, -3.0, 0.5, 0.20, 0.02),
    (9, -3.0, 3.0, 0.5, 0.20, 0.02),
    # Late-stage points
    (10, 3.0, 3.5, 0.8, 1.70, 0.12),
    (11, -3.8, -2.8, 0.8, 1.50, 0.10),
    (12, -2.0, 4.5, 0.8, 0.30, 0.05),
    # Boundary proximity / Outliers
    (13, 4.5, 0.0, 0.0, 0.10, 0.05),
    (14, 4.5, 0.0, 0.45, 0.40, 0.08),
    (15, 4.5, 0.0, 0.9, 0.50, 0.05),
]


@pytest.mark.parametrize("i, x, y, t, atol, rtol", TEST_POINTS)
def test_fdm_vs_mc_agreement(shared_eval_setup, i, x, y, t, atol, rtol):
    """Tests that the FDM solver matches Monte Carlo rollouts for specific states."""
    fdm_solver, V_sol, vec_env, policy, env = shared_eval_setup

    if x**2 + y**2 >= env.radius**2:
        pytest.skip(f"Point ({x}, {y}) is outside the environment boundary.")

    N_mc = 1024
    v_fdm = get_fdm_value(fdm_solver, V_sol, x, y, t)
    mc_mean, mc_std = run_mc_eval(
        vec_env, policy, t0=t, x0=np.array([x, y]), seed=42, N_mc=N_mc
    )

    abs_diff = abs(v_fdm - mc_mean)

    if abs(mc_mean) < 1e-8:
        rel_diff = 0.0 if abs_diff < 1e-8 else float("inf")
    else:
        rel_diff = abs_diff / abs(mc_mean)

    passed = (abs_diff <= atol) and (rel_diff <= rtol)

    error_msg = (
        f"\nMismatch at pt {i} (x={x}, y={y}, t={t}):\n"
        f"  FDM Value: {v_fdm:.4f}\n"
        f"  MC Value:  {mc_mean:.4f}\n"
        f"  Abs Diff:  {abs_diff:.4f} (Allowed: <= {atol})\n"
        f"  Rel Diff:  {rel_diff:.2%} (Allowed: <= {rtol:.1%})"
    )

    assert passed, error_msg


# --- DEBUG RUNNER ---

if __name__ == "__main__":
    print("\n" + "=" * 95)
    print(" FDM SOLVER vs MONTE CARLO DEBUG RUNNER ".center(95, "="))
    print("=" * 95 + "\n")

    # 1. Setup
    fdm_solver, V_sol, vec_env, policy, env = setup_evaluation_environment(verbose=True)
    N_mc = 1024

    # 2. Run Evaluation
    print("\nEvaluating Test Points:")
    print(
        f"pt idx | {'State (x,y)':<13} | {'Time':<5} | {'FDM':<8} | {'MC Mean +/- SE':<15} | {'Abs Diff (=<atol)':<18} | {'Rel Diff (=<rtol)':<18} | {'Status'}"
    )
    print("-" * 95)

    for i, x, y, t, atol, rtol in TEST_POINTS:
        if x**2 + y**2 >= env.radius**2:
            continue

        v_fdm = get_fdm_value(fdm_solver, V_sol, x, y, t)
        mc_mean, mc_std = run_mc_eval(
            vec_env, policy, t0=t, x0=np.array([x, y]), seed=42, N_mc=N_mc
        )
        mc_stderr = mc_std / np.sqrt(N_mc)

        abs_diff = abs(v_fdm - mc_mean)
        if abs(mc_mean) < 1e-8:
            rel_diff = 0.0 if abs_diff < 1e-8 else float("inf")
        else:
            rel_diff = abs_diff / abs(mc_mean)

        passed = (abs_diff <= atol) and (rel_diff <= rtol)
        status = "PASS" if passed else "FAIL"

        # Format columns
        coord_str = f"({x:>4.1f},{y:>4.1f})"
        mc_str = f"{mc_mean:>6.2f} +/-{mc_stderr:<4.2f}"

        # Include tolerances directly in the difference outputs
        abs_str = f"{abs_diff:>5.2f} (=<{atol:.2f})"
        if rel_diff != float("inf"):
            rel_str = f"{rel_diff:>6.1%} (=<{rtol:.0%})"
        else:
            rel_str = f"  inf% (=<{rtol:.0%})"

        print(
            f"pt {i} | {coord_str:<13} | {t:>4.2f}  | {v_fdm:>8.2f} | {mc_str:<15} | {abs_str:<18} | {rel_str:<18} | {status}"
        )

    print("-" * 95)
    vec_env.close()
