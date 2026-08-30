import os
import argparse
import pytest
import numpy as np
import torch
import time

from sde_md.envs.sde_env import SDEDiffusionEnvDiscAct
from sde_md.policies.quantizedtabularpolicy import QuantizedTabularPolicy
from sde_md.solvers.fdm_solver import FDMValueEvaluator

# --- CONFIGURATION ---

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

CONFIGS_TO_TEST = [
    {"grid_bins": 41, "dt": 0.05},
    {"grid_bins": 81, "dt": 0.01},
]


def get_ref_filename(bins, dt, proposed=False):
    """Generates a standardized filename for the tensor."""
    suffix = "_proposed" if proposed else ""
    return os.path.join(FIXTURES_DIR, f"fdm_ref_b{bins}_dt{dt}{suffix}.npy")


# --- SHARED SETUP & LOGIC ---


def init_heuristic_tabular_policy(policy, env, fdm_solver):
    """Directly writes extreme logits into the QuantizedTabularPolicy."""
    u_x_grid, u_y_grid = np.array(fdm_solver.u_x_grid), np.array(fdm_solver.u_y_grid)
    max_x, max_y = np.max(np.abs(u_x_grid)), np.max(np.abs(u_y_grid))
    n_x, n_y = len(u_x_grid), len(u_y_grid)

    new_logits = np.zeros_like(
        policy.tabular_params["action_logits"].detach().cpu().numpy()
    )

    for k in range(fdm_solver.time_steps + 1):
        t = k * fdm_solver.dt
        r = np.sqrt(fdm_solver.flat_x**2 + fdm_solver.flat_y**2)
        r_safe = np.maximum(r, 1e-8)

        scale = np.clip(
            (np.minimum(r, fdm_solver.R) / fdm_solver.R) * (t / fdm_solver.T), 0.0, 1.0
        )

        dir_x = np.where(r == 0, 0.0, -fdm_solver.flat_x / r_safe)
        dir_y = np.where(r == 0, 0.0, -fdm_solver.flat_y / r_safe)

        ideal_x, ideal_y = dir_x * scale * max_x, dir_y * scale * max_y

        idx_x = np.argmin(np.abs(ideal_x[:, None] - u_x_grid[None, :]), axis=1)
        idx_y = np.argmin(np.abs(ideal_y[:, None] - u_y_grid[None, :]), axis=1)

        obs_batch = np.stack(
            [fdm_solver.flat_x, fdm_solver.flat_y, np.full_like(fdm_solver.flat_x, t)],
            axis=1,
        )
        obs_tensor = torch.as_tensor(
            obs_batch, dtype=torch.float32, device=policy.device
        )
        policy_indices = policy._quantize(obs_tensor).cpu().numpy()

        time_slice_logits = np.full(
            (fdm_solver.n_states, n_x + n_y), -100.0, dtype=np.float32
        )
        time_slice_logits[np.arange(fdm_solver.n_states), idx_x] = 100.0
        time_slice_logits[np.arange(fdm_solver.n_states), n_x + idx_y] = 100.0
        new_logits[policy_indices] = time_slice_logits

    with torch.no_grad():
        policy.tabular_params["action_logits"].copy_(
            torch.tensor(new_logits, device=policy.device)
        )
    policy.normalize_policy_(nvec=getattr(policy.action_space, "nvec", None))


def build_and_solve_fdm(grid_bins, dt):
    """Initializes the environment and policy, then runs the FDM solver."""
    env_kwargs = {
        "radius": 5.0,
        "sigma": 0.05,
        "T": 1.0,
        "dt": dt,
        "action_min": -15.0,
        "action_max": 15.0,
        "n_actions_per_dim": 5,
        "state_dim": 2,
        "action_dim": 2,
        "control_cost_factor": 0.1,
    }
    env = SDEDiffusionEnvDiscAct(**env_kwargs)

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

    fdm_solver = FDMValueEvaluator(
        env=env, policy=policy, grid_bins=grid_bins, ent_coef=1e-6
    )
    init_heuristic_tabular_policy(policy, env, fdm_solver)

    V_sol = fdm_solver.solve()
    return fdm_solver, V_sol


def calculate_regression_error(bins, dt):
    """Shared evaluation logic: Loads reference, runs solver, calculates L-infinity."""
    ref_file = get_ref_filename(bins, dt, proposed=False)

    if not os.path.exists(ref_file):
        return None, f"Reference file missing: {os.path.basename(ref_file)}"

    V_ref = np.load(ref_file)
    _, V_sol = build_and_solve_fdm(bins, dt)

    l_inf_norm = float(np.max(np.abs(V_sol - V_ref)))
    return l_inf_norm, None


# --- PYTEST LOGIC ---


@pytest.mark.parametrize("config", CONFIGS_TO_TEST)
def test_fdm_tensor_regression(config):
    """Pytest entry point for continuous integration."""
    bins = config["grid_bins"]
    dt = config["dt"]
    tol = 1e-5

    l_inf_norm, error_msg = calculate_regression_error(bins, dt)

    if error_msg:
        pytest.skip(error_msg + " (Run script with --generate first)")

    assert (
        l_inf_norm < tol
    ), f"Regression detected! L-infinity norm is {l_inf_norm:.6e} (Allowed: < {tol})."


# --- CLI DEBUG RUNNER & GENERATOR ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDM Tensor Regression Utility")
    parser.add_argument(
        "--generate", action="store_true", help="Generate proposed reference tensors"
    )
    args = parser.parse_args()

    if args.generate:
        print("\n" + "=" * 75)
        print(" GENERATING PROPOSED 3D FDM TENSORS ".center(75, "="))
        print("=" * 75)

        for config in CONFIGS_TO_TEST:
            bins = config["grid_bins"]
            dt = config["dt"]

            print(f"\n[*] Solving for grid_bins={bins}, dt={dt}...")
            start_t = time.time()
            _, V_sol = build_and_solve_fdm(bins, dt)
            print(f"    -> Solved in {time.time() - start_t:.2f}s")

            proposed_path = get_ref_filename(bins, dt, proposed=True)
            np.save(proposed_path, V_sol)
            print(
                f"    -> Saved tensor of shape {V_sol.shape} to {os.path.basename(proposed_path)}"
            )

        print(f"\nAll proposed tensors generated in {FIXTURES_DIR}/")
        print(
            "Remove the '_proposed' suffix from the filenames to make them the official test targets.\n"
        )

    else:
        # Default behavior with no arguments: Run the debug checker
        print("\n" + "=" * 75)
        print(" FDM REGRESSION DEBUG RUNNER ".center(75, "="))
        print("=" * 75 + "\n")

        tol = 1e-5

        print(
            f"{'Configuration':<25} | {'L-inf Error':<15} | {'Tolerance':<15} | {'Status'}"
        )
        print("-" * 75)

        for config in CONFIGS_TO_TEST:
            bins = config["grid_bins"]
            dt = config["dt"]
            config_str = f"bins={bins}, dt={dt}"

            l_inf_norm, error_msg = calculate_regression_error(bins, dt)

            if error_msg:
                print(
                    f"{config_str:<25} | {'MISSING FILE':<15} | {f'<{tol}':<15} | SKIP"
                )
                continue

            passed = l_inf_norm < tol
            status = "PASS" if passed else "FAIL"

            print(
                f"{config_str:<25} | {l_inf_norm:<15.2e} | {f'<{tol}':<15} | {status}"
            )

        print("-" * 75 + "\n")
        print(
            "Run with `--generate` if you need to create or update the golden references."
        )
