import pytest
import numpy as np
import torch
import time

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from sde_md.envs.sde_env import SDEDiffusionEnv
from sde_md.policies.quantizedtabularzpolicy import QuantizedTabularZPolicy
from sde_md.policies.projected_lqr_policy import ProjectedLQRPolicy
from sde_md.solvers.cont_act_fdm_solver import ContinuousActionsFDMValueEvaluator
from sde_md.iterators.dpp_based_iterations import run_mc_eval


def init_continuous_heuristic_policy(policy, env, fdm_solver):
    """Directly writes values into the Z-table of the QuantizedTabularZPolicy."""
    Z_table = np.zeros_like(policy.tabular_params["Z_table"].detach().cpu().numpy())

    for k in range(fdm_solver.time_steps + 1):
        t = k * fdm_solver.dt
        r = np.sqrt(fdm_solver.flat_x**2 + fdm_solver.flat_y**2)
        r_safe = np.maximum(r, 1e-8)
        scale = (np.minimum(r, fdm_solver.R) / fdm_solver.R) * (t / fdm_solver.T)
        scale = np.clip(scale, 0.0, 1.0)
        z_mag = 5.0 * scale

        Z_x = -(fdm_solver.flat_x / r_safe) * z_mag
        Z_y = -(fdm_solver.flat_y / r_safe) * z_mag

        obs_batch = np.stack(
            [fdm_solver.flat_x, fdm_solver.flat_y, np.full_like(fdm_solver.flat_x, t)],
            axis=1,
        )
        obs_tensor = torch.as_tensor(
            obs_batch, dtype=torch.float32, device=policy.device
        )
        policy_indices = policy._quantize(obs_tensor).cpu().numpy()

        Z_table[policy_indices, 0] = Z_x
        Z_table[policy_indices, 1] = Z_y

    with torch.no_grad():
        policy.tabular_params["Z_table"].copy_(
            torch.tensor(Z_table, device=policy.device)
        )


def get_fdm_value(fdm_solver, V_sol, x, y, t):
    t_safe = min(max(t, 0.0), fdm_solver.T)
    t_idx = int(np.round(t_safe / fdm_solver.dt))
    x_idx = np.argmin(np.abs(fdm_solver.x_axis - x))
    y_idx = np.argmin(np.abs(fdm_solver.y_axis - y))
    return V_sol[t_idx, x_idx, y_idx]


# --- SHARED SETUP LOGIC ---


def setup_z_evaluation_environment(n_envs: int, verbose=False):
    if verbose:
        print("\n[Z-Policy 1/3] Initializing Continuous Environment and Z-Policy...")
    env_kwargs = {
        "env_type": "continuous",
        "radius": 5.0,
        "sigma": 0.05,
        "T": 1.0,
        "dt": 0.01,
        "action_bound": 15.0,
        "state_dim": 2,
        "action_dim": 2,
        "control_cost_factor": 0.1,
        "space_cost_factor_terminal": 2.0,
    }
    env = SDEDiffusionEnv(**env_kwargs)
    grid_bins = 161
    policy = QuantizedTabularZPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.0,
        grid_bins=grid_bins,
        radius=env_kwargs["radius"],
        T=env_kwargs["T"],
        dt=env_kwargs["dt"],
        action_radius=env_kwargs["action_bound"],
    )
    policy.eval()

    if verbose:
        print(
            f"[Z-Policy 2/3] Solving Continuous FDM on {grid_bins}x{grid_bins} grid..."
        )
        start_time = time.time()

    fdm_solver = ContinuousActionsFDMValueEvaluator(
        env=env,
        policy=policy,
        grid_bins=grid_bins,
        ent_coef=0.0,
        action_radius=env_kwargs["action_bound"],
    )
    init_continuous_heuristic_policy(policy, env, fdm_solver)
    V_sol = fdm_solver.solve()

    if verbose:
        print(f"      -> Done in {time.time() - start_time:.2f} seconds.")
        print(f"[Z-Policy 3/3] Launching parallel environments for MC rollouts...")

    vec_env = make_vec_env(
        SDEDiffusionEnv, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv
    )
    return fdm_solver, V_sol, vec_env, policy, env


def setup_lqr_evaluation_environment(n_envs: int, verbose=False):
    if verbose:
        print(
            "\n[LQR-Policy 1/3] Initializing Continuous Environment and LQR-Policy..."
        )
    env_kwargs = {
        "env_type": "continuous",
        "radius": 5.0,
        "sigma": 0.05,
        "T": 1.0,
        "dt": 0.01,
        "action_bound": 15.0,
        "state_dim": 2,
        "action_dim": 2,
        "control_cost_factor": 0.1,
        "space_cost_factor_terminal": 2.0,
    }
    env = SDEDiffusionEnv(**env_kwargs)
    grid_bins = 161

    policy = ProjectedLQRPolicy(
        T=env_kwargs["T"],
        action_radius=env_kwargs["action_bound"],
        control_cost=env_kwargs["control_cost_factor"],
        terminal_cost=env_kwargs["space_cost_factor_terminal"],
        sigma=env_kwargs["sigma"],
        state_dim=env_kwargs["state_dim"],
    )

    if verbose:
        print(
            f"[LQR-Policy 2/3] Solving Continuous FDM on {grid_bins}x{grid_bins} grid..."
        )
        start_time = time.time()

    fdm_solver = ContinuousActionsFDMValueEvaluator(
        env=env,
        policy=policy,
        grid_bins=grid_bins,
        ent_coef=0.0,
        action_radius=env_kwargs["action_bound"],
    )
    V_sol = fdm_solver.solve()

    if verbose:
        print(f"      -> Done in {time.time() - start_time:.2f} seconds.")
        print(f"[LQR-Policy 3/3] Launching parallel environments for MC rollouts...")

    vec_env = make_vec_env(
        SDEDiffusionEnv, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv
    )
    return fdm_solver, V_sol, vec_env, policy, env


# --- PYTEST CONFIGURATION & TESTS ---


@pytest.fixture(scope="module")
def shared_z_setup():
    fdm_solver, V_sol, vec_env, policy, env = setup_z_evaluation_environment(
        n_envs=16, verbose=False
    )
    yield fdm_solver, V_sol, vec_env, policy, env
    vec_env.close()


@pytest.fixture(scope="module")
def shared_lqr_setup():
    fdm_solver, V_sol, vec_env, policy, env = setup_lqr_evaluation_environment(
        n_envs=16, verbose=False
    )
    yield fdm_solver, V_sol, vec_env, policy, env
    vec_env.close()


TEST_POINTS_LQR = [
    # Origin
    (1, 0.2, 0.1, 0.0, 0.02, 1.2),
    (2, -0.2, 0.05, 0.5, 0.05, 0.80),
    (3, 0.15, 0.05, 0.9, 0.03, 0.40),
    # Mid-range points
    (4, 3.0, 3.0, 0.0, 0.28, 0.16),
    (5, 3.0, -3.2, 0.0, 0.17, 0.10),
    (6, -3.0, 2.8, 0.0, 0.17, 0.11),
    (7, 3.0, 3.0, 0.5, 0.35, 0.11),
    (8, 3.0, -3.0, 0.5, 0.32, 0.10),
    (9, -3.0, 3.0, 0.5, 0.33, 0.10),
    # Late-stage points
    (10, 3.0, 3.5, 0.8, 1.70, 0.13),
    (11, -3.8, -2.8, 0.8, 1.40, 0.14),
    (12, -2.0, 4.5, 0.8, 1.30, 0.11),
    # Boundary proximity / Outliers
    (13, 4.5, 0.0, 0.0, 0.15, 0.08),
    (14, 4.5, 0.0, 0.45, 0.40, 0.09),
    (15, 4.5, 0.0, 0.9, 0.65, 0.05),
]

TEST_POINTS_Z = [
    # Origin
    (1, 0.2, 0.1, 0.0, 0.05, 0.35),
    (2, -0.2, 0.05, 0.5, 0.05, 0.25),
    (3, 0.15, 0.05, 0.9, 0.03, 0.07),
    # Mid-range points
    (4, 3.0, 3.0, 0.0, 0.19, 0.04),
    (5, 3.0, -3.0, 0.0, 0.07, 0.02),
    (6, -3.0, 3.0, 0.0, 0.06, 0.02),
    (7, 3.0, 3.0, 0.5, 0.06, 0.02),
    (8, 3.0, -3.0, 0.5, 0.06, 0.02),
    (9, -3.0, 3.0, 0.5, 0.06, 0.02),
    # Late-stage points
    (10, 3.0, 3.5, 0.8, 1.30, 0.13),
    (11, -3.8, -2.8, 0.8, 1.40, 0.13),
    (12, -2.0, 4.5, 0.8, 1.25, 0.11),
    # Boundary proximity / Outliers
    (13, 4.5, 0.0, 0.0, 0.10, 0.05),
    (14, 4.5, 0.0, 0.45, 0.40, 0.08),
    (15, 4.5, 0.0, 0.9, 0.65, 0.05),
]


def run_agreement_test(setup_tuple, i, x, y, t, atol, rtol):
    """Helper to run the numerical comparison logic for any policy setup."""
    fdm_solver, V_sol, vec_env, policy, env = setup_tuple

    if x**2 + y**2 >= env.radius**2:
        pytest.skip(f"Point ({x}, {y}) is outside the environment boundary.")

    v_fdm = get_fdm_value(fdm_solver, V_sol, x, y, t)
    mc_mean, _ = run_mc_eval(
        vec_env, policy, t0=t, x0=np.array([x, y]), seed=42, N_mc=1024
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


@pytest.mark.parametrize("i, x, y, t, atol, rtol", TEST_POINTS_Z)
def test_cont_z_fdm_vs_mc_agreement(shared_z_setup, i, x, y, t, atol, rtol):
    """Tests agreement using the QuantizedTabularZPolicy."""
    run_agreement_test(shared_z_setup, i, x, y, t, atol, rtol)


@pytest.mark.parametrize("i, x, y, t, atol, rtol", TEST_POINTS_LQR)
def test_cont_lqr_fdm_vs_mc_agreement(shared_lqr_setup, i, x, y, t, atol, rtol):
    """Tests agreement using the ProjectedLQRPolicy."""
    run_agreement_test(shared_lqr_setup, i, x, y, t, atol, rtol)


# --- DEBUG RUNNER ---


def run_debug_evaluation(name, setup_tuple, test_points):
    fdm_solver, V_sol, vec_env, policy, env = setup_tuple
    N_mc = 1024

    print(f"\nEvaluating Test Points for {name}:")
    print(
        f"pt idx | {'State (x,y)':<13} | {'Time':<5} | {'FDM':<8} | {'MC Mean +/- SE':<15} | {'Abs Diff (=<atol)':<18} | {'Rel Diff (=<rtol)':<18} | {'Status'}"
    )
    print("-" * 95)

    for i, x, y, t, atol, rtol in test_points:
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

        coord_str = f"({x:>4.1f},{y:>4.1f})"
        mc_str = f"{mc_mean:>6.2f} +/-{mc_stderr:<4.2f}"
        abs_str = f"{abs_diff:>5.2f} (=<{atol:.2f})"
        rel_str = (
            f"{rel_diff:>6.1%} (=<{rtol:.0%})"
            if rel_diff != float("inf")
            else f"  inf% (=<{rtol:.0%})"
        )

        print(
            f"pt {i}  | {coord_str:<13} | {t:>4.2f}  | {v_fdm:>8.2f} | {mc_str:<15} | {abs_str:<18} | {rel_str:<18} | {status}"
        )

    print("-" * 95)
    vec_env.close()


if __name__ == "__main__":
    print("\n" + "=" * 95)
    print(" CONTINUOUS FDM SOLVER vs MONTE CARLO DEBUG RUNNER ".center(95, "="))
    print("=" * 95)

    # Run LQR-Policy Evaluation
    lqr_setup = setup_lqr_evaluation_environment(n_envs=16, verbose=True)
    run_debug_evaluation("ProjectedLQRPolicy", lqr_setup, test_points=TEST_POINTS_LQR)

    # Run Z-Policy Evaluation
    z_setup = setup_z_evaluation_environment(n_envs=16, verbose=True)
    run_debug_evaluation("QuantizedTabularZPolicy", z_setup, test_points=TEST_POINTS_Z)
