import os
import argparse
import pytest
import numpy as np
import torch
import time
import itertools

from sde_md.envs.sde_env import SDEDiffusionEnv
from sde_md.policies.quantizedtabularzpolicy import QuantizedTabularZPolicy
from sde_md.policies.projected_lqr_policy import ProjectedLQRPolicy
from sde_md.solvers.cont_act_fdm_solver import ContinuousActionsFDMValueEvaluator

# --- CONFIGURATION ---

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

POLICIES_TO_TEST = ["ZPolicy", "LQR"]

CONFIGS_TO_TEST = [
    {"grid_bins": 41, "dt": 0.05},
    {"grid_bins": 81, "dt": 0.01},
]


def get_ref_filename(policy_name, bins, dt, proposed=False):
    """Generates a standardized filename for the continuous tensor."""
    suffix = "_proposed" if proposed else ""
    return os.path.join(
        FIXTURES_DIR, f"fdm_cont_ref_{policy_name}_b{bins}_dt{dt}{suffix}.npy"
    )


# --- SHARED SETUP & LOGIC ---


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


def build_and_solve_cont_fdm(policy_name, grid_bins, dt):
    """Initializes the continuous environment and the specified policy, then runs the FDM solver."""
    env_kwargs = {
        "env_type": "continuous",
        "radius": 5.0,
        "sigma": 0.05,
        "T": 1.0,
        "dt": dt,
        "action_bound": 15.0,
        "state_dim": 2,
        "action_dim": 2,
        "control_cost_factor": 0.1,
        "space_cost_factor_terminal": 2.0,
    }
    env = SDEDiffusionEnv(**env_kwargs)

    if policy_name == "ZPolicy":
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
        fdm_solver = ContinuousActionsFDMValueEvaluator(
            env=env,
            policy=policy,
            grid_bins=grid_bins,
            ent_coef=0.0,
            action_radius=env_kwargs["action_bound"],
        )
        init_continuous_heuristic_policy(policy, env, fdm_solver)

    elif policy_name == "LQR":
        policy = ProjectedLQRPolicy(
            T=env_kwargs["T"],
            action_radius=env_kwargs["action_bound"],
            control_cost=env_kwargs["control_cost_factor"],
            terminal_cost=env_kwargs["space_cost_factor_terminal"],
            sigma=env_kwargs["sigma"],
            state_dim=env_kwargs["state_dim"],
        )
        fdm_solver = ContinuousActionsFDMValueEvaluator(
            env=env,
            policy=policy,
            grid_bins=grid_bins,
            ent_coef=0.0,
            action_radius=env_kwargs["action_bound"],
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    V_sol = fdm_solver.solve()
    return fdm_solver, V_sol


def calculate_regression_error(policy_name, bins, dt):
    """Shared evaluation logic: Loads reference, runs solver, calculates L-infinity."""
    ref_file = get_ref_filename(policy_name, bins, dt, proposed=False)

    if not os.path.exists(ref_file):
        return None, f"Reference missing: {os.path.basename(ref_file)}"

    V_ref = np.load(ref_file)
    _, V_sol = build_and_solve_cont_fdm(policy_name, bins, dt)

    l_inf_norm = float(np.max(np.abs(V_sol - V_ref)))
    return l_inf_norm, None


# --- PYTEST LOGIC ---

# Create a flat list of test parameters (e.g., [("ZPolicy", 41, 0.05), ...])
TEST_MATRIX = [
    (policy, config["grid_bins"], config["dt"])
    for policy in POLICIES_TO_TEST
    for config in CONFIGS_TO_TEST
]


@pytest.mark.parametrize("policy_name, bins, dt", TEST_MATRIX)
def test_cont_fdm_tensor_regression(policy_name, bins, dt):
    """Pytest entry point for continuous integration."""
    tol = 1e-5

    l_inf_norm, error_msg = calculate_regression_error(policy_name, bins, dt)

    if error_msg:
        pytest.skip(error_msg + " (Run script with --generate first)")

    assert (
        l_inf_norm < tol
    ), f"[{policy_name}] Regression detected! L-infinity norm is {l_inf_norm:.6e} (Allowed: < {tol})."


# --- CLI DEBUG RUNNER & GENERATOR ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous FDM Tensor Regression Utility"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate proposed reference tensors"
    )
    args = parser.parse_args()

    if args.generate:
        print("\n" + "=" * 80)
        print(" GENERATING PROPOSED 3D CONTINUOUS FDM TENSORS ".center(80, "="))
        print("=" * 80)

        for policy_name in POLICIES_TO_TEST:
            print(f"\n>>> Running for Policy: {policy_name}")
            for config in CONFIGS_TO_TEST:
                bins = config["grid_bins"]
                dt = config["dt"]

                print(f"[*] Solving for grid_bins={bins}, dt={dt}...")
                start_t = time.time()
                _, V_sol = build_and_solve_cont_fdm(policy_name, bins, dt)
                print(f"    -> Solved in {time.time() - start_t:.2f}s")

                proposed_path = get_ref_filename(policy_name, bins, dt, proposed=True)
                np.save(proposed_path, V_sol)
                print(
                    f"    -> Saved {V_sol.shape} tensor to {os.path.basename(proposed_path)}"
                )

        print(f"\n All proposed continuous tensors generated in {FIXTURES_DIR}/")
        print(
            "Remove the '_proposed' suffix from the filenames to make them the official test targets.\n"
        )

    else:
        # Default behavior with no arguments: Run the debug checker
        print("\n" + "=" * 85)
        print(" CONTINUOUS FDM REGRESSION DEBUG RUNNER ".center(85, "="))
        print("=" * 85 + "\n")

        tol = 1e-5

        print(
            f"{'Policy':<12} | {'Configuration':<22} | {'L-inf Error':<15} | {'Tolerance':<12} | {'Status'}"
        )
        print("-" * 85)

        for policy_name in POLICIES_TO_TEST:
            for config in CONFIGS_TO_TEST:
                bins = config["grid_bins"]
                dt = config["dt"]
                config_str = f"b={bins}, dt={dt}"

                l_inf_norm, error_msg = calculate_regression_error(
                    policy_name, bins, dt
                )

                if error_msg:
                    print(
                        f"{policy_name:<12} | {config_str:<22} | {'MISSING FILE':<15} | {f'<{tol}':<12} | SKIP"
                    )
                    continue

                passed = l_inf_norm < tol
                status = "PASS" if passed else "FAIL"

                print(
                    f"{policy_name:<12} | {config_str:<22} | {l_inf_norm:<15.2e} | {f'<{tol}':<12} | {status}"
                )

        print("-" * 85 + "\n")
        print(
            "Run with `--generate` if you need to create or update the golden references."
        )
