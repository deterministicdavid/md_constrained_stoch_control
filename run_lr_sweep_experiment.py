"""
Learning Rate Sweep Experiment.

Evaluates how different learning rates (e.g., `md_lambda`, `lr`, `pg_lr`, `alpha`)
affect policy performance after a fixed number of iterations across one or
more algorithm configurations.

Performance is evaluated via fixed-seed Monte Carlo rollouts in the SDE environment.
The resulting metrics (MC Return and MC Cost = -Return) are saved to a CSV file.
It groups results by the `env_config` used, producing one MC Cost plot per environment.

Usage:
    python run_lr_sweep_experiment.py --config lr_sweep_config.yaml

Re-plot from existing CSV without re-running computation:
    python run_lr_sweep_experiment.py --plot_csv results_lr_sweep/lr_sweep_results.csv
"""

import argparse
import copy
import csv
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import numpy as np
import torch
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from run_dpp_based import load_yaml, setup_env_and_policy
from sde_md.envs.sde_env import SDEDiffusionEnv, SDEDiffusionEnvDiscAct

from sde_md.iterators.dpp_based_iterations import run_mc_eval
from sde_md.iterators.dpp_based_cont_act_iterations import (
    ContinuousActionMirrorDescent,
)
from sde_md.iterators.dpp_based_cont_act_proj_grad import (
    ContinuousActionProjectedGradient,
)
from sde_md.iterators.dpp_based_soft_iterations import (
    ExactMirrorDescent,
    ProjectedProbabilityDescent,
    SoftPolicyIteration,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def create_solver(env, policy, dpp_cfg, grid_bins, ent_coef):
    """Factory helper to instantiate the correct solver class."""
    algo = dpp_cfg.get("algo", "spi")
    if algo == "spi":
        return SoftPolicyIteration(
            env, policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef
        )
    elif algo == "exact_md":
        return ExactMirrorDescent(
            env, policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef
        )
    elif algo == "projected_pg":
        return ProjectedProbabilityDescent(
            env, policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef
        )
    elif algo == "cont_act_md":
        action_radius = dpp_cfg.get("algo_kwargs", {}).get("action_radius", 1.0)
        return ContinuousActionMirrorDescent(
            env=env,
            policy=policy,
            grid_bins=grid_bins,
            gamma=dpp_cfg["gamma"],
            ent_coef=ent_coef,
            action_radius=action_radius,
        )
    elif algo == "cont_act_proj_pg":
        action_radius = dpp_cfg.get("algo_kwargs", {}).get("action_radius", 1.0)
        return ContinuousActionProjectedGradient(
            env=env,
            policy=policy,
            grid_bins=grid_bins,
            gamma=dpp_cfg["gamma"],
            ent_coef=ent_coef,
            action_radius=action_radius,
        )
    else:
        raise ValueError(f"Unknown DPP algorithm: {algo}")


def _worker_run_lr(args):
    """Module-level worker function for multiprocessing."""
    exp_name, param_name, param_val, env_config_path, env_cfg, dpp_cfg, n_iterations, mc_cfg = args
    t_start = time.time()

    # Create fresh env and policy for this parameter trial
    env, policy, _, _, _ = setup_env_and_policy(env_cfg, dpp_cfg)
    grid_bins = dpp_cfg["grid_bins"]
    ent_coef = float(dpp_cfg["ent_coef"])

    solver = create_solver(env, policy, dpp_cfg, grid_bins, ent_coef)

    # Prepare iteration keyword arguments
    iter_kwargs = {param_name: param_val}
    solver.iterate(n_iterations=n_iterations, **iter_kwargs)

    # --- MC Evaluation using imported run_mc_eval ---
    env_type = env_cfg.get("env_type", "discrete")
    env_class = SDEDiffusionEnv if env_type == "continuous" else SDEDiffusionEnvDiscAct
    env_kwargs = {k: v for k, v in env_cfg.items() if k != "env_type"}

    vec_env = make_vec_env(
        env_class,
        n_envs=mc_cfg["n_envs"],
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )
    
    # Imported run_mc_eval expects t0 and x0 (None for default start)

    start_radius = 4.8
    theta = np.pi / 4
    t0 = 0.0
    x0 = np.array(
        [start_radius * np.cos(theta), start_radius * np.sin(theta)],
        dtype=np.float32,
    )

    mean_ret, std_ret = run_mc_eval(
        vec_env=vec_env,
        policy=policy,
        t0=t0,
        x0=x0,
        seed=mc_cfg["seed"],
        N_mc=mc_cfg["N_mc"],
    )
    vec_env.close()
    
    # Calculate Standard Error (SE) from Standard Deviation
    se_ret = float(std_ret) / math.sqrt(mc_cfg["N_mc"])
    mean_ret = float(mean_ret)

    wall_time = time.time() - t_start

    result = {
        "exp_name": exp_name,
        "env_config": env_config_path,
        "algo": dpp_cfg.get("algo", "unknown"),
        "param_name": param_name,
        "param_value": float(param_val),
        "n_iterations": n_iterations,
        "mc_return": mean_ret,
        "mc_return_se": se_ret,
        "mc_cost": -mean_ret,
        "mc_cost_se": se_ret,
        "wall_time_s": wall_time,
    }
    return result


def write_csv(rows, csv_path):
    fieldnames = [
        "exp_name",
        "env_config",
        "algo",
        "param_name",
        "param_value",
        "n_iterations",
        "mc_cost",
        "mc_cost_se",
        "mc_return",
        "mc_return_se",
        "wall_time_s",
    ]
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(rows, output_dir):
    """
    Plots Cost vs Learning Rate, clustered by Environment Config.
    Generates:
      1) Individual plots per environment (independent y-axis).
      2) A single combined figure with a 2-column layout and a shared y-axis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify unique environments
    env_configs = sorted(list({r["env_config"] for r in rows}))
    n_envs = len(env_configs)
    
    if n_envs == 0:
        print("No data to plot.")
        return

    # Determine combined layout: 2 columns, dynamic rows
    cols = min(2, n_envs)
    rows_layout = math.ceil(n_envs / cols)
    
    # Create the combined figure with shared y-axis
    fig_all, axes_all = plt.subplots(
        rows_layout, cols, 
        figsize=(6 * cols, 4.5 * rows_layout), 
        sharey=True, 
        squeeze=False # Ensures axes is always a 2D array
    )
    axes_all_flat = axes_all.flatten()

    for idx, env_path in enumerate(env_configs):
        ax_all = axes_all_flat[idx]
        
        # Create an individual figure for this environment (independent y-axis)
        fig_single, ax_single = plt.subplots(figsize=(8, 5))
        
        # Filter rows for this specific environment
        env_rows = [r for r in rows if r["env_config"] == env_path]
        exp_names = sorted(list({r["exp_name"] for r in env_rows}))

        for exp in exp_names:
            exp_rows_subset = [r for r in env_rows if r["exp_name"] == exp]
            exp_rows_subset.sort(key=lambda r: float(r["param_value"]))

            xs = [float(r["param_value"]) for r in exp_rows_subset]
            costs = [float(r["mc_cost"]) for r in exp_rows_subset]
            cost_errs = [float(r["mc_cost_se"]) * 1.96 for r in exp_rows_subset]  # 95% CI
            param_name = exp_rows_subset[0]["param_name"]

            # Plot on BOTH the combined subplot and the individual figure
            for ax in (ax_all, ax_single):
                line = ax.errorbar(
                    xs,
                    costs,
                    yerr=cost_errs,
                    marker="o",
                    capsize=3,
                    label=f"{exp} ({param_name})",
                )
                
                # Highlight best (minimum cost)
                best_cost_idx = np.argmin(costs)
                ax.plot(
                    xs[best_cost_idx],
                    costs[best_cost_idx],
                    marker="*",
                    color=line[0].get_color(),
                    markersize=12,
                    zorder=5,
                )

        env_basename = os.path.splitext(os.path.basename(env_path))[0]
        safe_env_name = env_basename.replace("/", "_").replace("\\", "_")

        # --- Formatting & Saving the INDIVIDUAL Subplot ---
        ax_single.set_xscale("log")
        ax_single.set_xlabel("Learning Rate / Parameter (log scale)")
        ax_single.set_ylabel("MC Cost ($-Return$)")
        ax_single.set_title(f"Environment: {env_basename}")
        ax_single.grid(True, which="both", alpha=0.3)
        ax_single.legend()

        fig_single.tight_layout()
        for ext in ("pdf", "png"):
            fig_single.savefig(
                os.path.join(output_dir, f"lr_sweep_cost_{safe_env_name}.{ext}"), 
                dpi=150, 
                bbox_inches="tight"
            )
        plt.close(fig_single)


        # --- Formatting the COMBINED Subplot ---
        ax_all.set_xscale("log")
        ax_all.set_xlabel("Learning Rate / Parameter (log scale)")
        
        # Only set Y-label on the left column to keep it clean
        if idx % cols == 0:
            ax_all.set_ylabel("MC Cost ($-Return$)")
        
        ax_all.set_title(f"Environment: {env_basename}")
        ax_all.grid(True, which="both", alpha=0.3)
        ax_all.legend()


    # --- Finalizing & Saving the COMBINED Plot ---
    # Hide any unused subplots if the number of environments is odd
    for idx in range(n_envs, len(axes_all_flat)):
        fig_all.delaxes(axes_all_flat[idx])

    # Add a global title and adjust layout
    fig_all.suptitle("Cost vs Learning Rate (Lower is Better)", fontsize=16, y=1.02)
    fig_all.tight_layout()
    
    # Save the single combined plot
    for ext in ("pdf", "png"):
        fig_all.savefig(
            os.path.join(output_dir, f"lr_sweep_cost_all_envs.{ext}"), 
            dpi=150, 
            bbox_inches="tight" # Prevents the suptitle from getting cut off
        )
    plt.close(fig_all)


def validate_and_load_experiments(experiments):
    """
    Pre-flight check: Loads all YAML files and verifies required keys exist
    BEFORE any heavy computation starts. Fails fast if anything is wrong.
    Also ensures critical parameters (dt, n_iterations, grid_bins, gamma) 
    are strictly identical across all experiments to guarantee a fair comparison.
    """
    print("Running pre-flight config checks...")
    validated_exps = []
    
    reference_params = None
    
    for i, exp in enumerate(experiments):
        exp_name = exp.get("name", f"Experiment_{i}")
        env_path = exp.get("env_config")
        dpp_path = exp.get("dpp_config")
        
        # 1. Check file existence
        if not env_path or not os.path.exists(env_path):
            raise FileNotFoundError(f"[{exp_name}] env_config missing or not found: {env_path}")
        if not dpp_path or not os.path.exists(dpp_path):
            raise FileNotFoundError(f"[{exp_name}] dpp_config missing or not found: {dpp_path}")
            
        # 2. Load YAMLs
        env_cfg = load_yaml(env_path)
        dpp_cfg = load_yaml(dpp_path)
        
        # 3. Check strictly required keys in DPP config
        required_dpp_keys = ["grid_bins", "ent_coef", "gamma", "algo", "dt"]
        for key in required_dpp_keys:
            if key not in dpp_cfg:
                raise KeyError(f"[{exp_name}] Required key '{key}' is missing in {dpp_path}")
                
        # 4. Check experiment definition parameters
        if "param_name" not in exp or "param_values" not in exp:
            raise KeyError(f"[{exp_name}] Sweep config must specify 'param_name' and 'param_values'")
            
        # 5. Cross-experiment consistency check
        current_params = {
            "dt": dpp_cfg.get("dt"),
            "grid_bins": dpp_cfg.get("grid_bins"),
            "gamma": dpp_cfg.get("gamma"),
        }
        
        if reference_params is None:
            reference_params = current_params
        else:
            for key, ref_val in reference_params.items():
                curr_val = current_params[key]
                if curr_val != ref_val:
                    raise ValueError(
                        f"[{exp_name}] Consistency Validation Failed!\n"
                        f"Parameter '{key}' must match across all experiments to ensure a fair sweep.\n"
                        f"Expected: {ref_val} (from first experiment)\n"
                        f"Got:      {curr_val} (in {dpp_path})"
                    )
            
        # Store loaded configs to avoid reloading them later
        exp["loaded_env_cfg"] = env_cfg
        exp["loaded_dpp_cfg"] = dpp_cfg
        validated_exps.append(exp)
        
    print("All configurations loaded and validated successfully!\n")
    return validated_exps


def main(args):
    # --- Plot-only Mode ---
    if args.plot_csv:
        output_dir = "results_lr_sweep" # Default fallback
        if args.config and os.path.exists(args.config):
             output_dir = load_yaml(args.config).get("output_dir", output_dir)
             
        numeric_keys = {
            "param_value",
            "n_iterations",
            "mc_cost",
            "mc_cost_se",
            "mc_return",
            "mc_return_se",
            "wall_time_s",
        }
        
        # Explicitly define the columns since older CSVs might not have a header row
        expected_fieldnames = [
            "exp_name", "env_config", "algo", "param_name", "param_value",
            "n_iterations", "mc_cost", "mc_cost_se", "mc_return", "mc_return_se", "wall_time_s"
        ]
        
        rows = []
        for path in args.plot_csv:
            with open(path, newline="") as f:
                # Pass fieldnames so it doesn't consume the first data row as a header
                reader = csv.DictReader(f, fieldnames=expected_fieldnames)
                
                for r in reader:
                    # If the file *did* have a header row, skip it
                    if r["exp_name"] == "exp_name":
                        continue
                        
                    # Safely parse numeric columns
                    for k, v in list(r.items()):
                        if k in numeric_keys and v not in (None, ""):
                            try:
                                r[k] = float(v)
                            except ValueError:
                                pass
                                
                    rows.append(r)
            print(f"Loaded {len(rows)} rows from {path}")

        plot_results(rows, output_dir)
        print(f"Plots saved to {output_dir}/")
        return
    
    # --- Standard Execution Mode ---
    if not args.config:
        raise ValueError("You must specify a --config file when not running in plot-only mode.")
        
    config = load_yaml(args.config)
    output_dir = config.get("output_dir", "results_lr_sweep")

    os.makedirs(output_dir, exist_ok=True)
    n_iterations = config.get("n_iterations", 50)
    mc_cfg = dict(config.get("mc_eval", {"n_envs": 16, "N_mc": 512, "seed": 42}))
    n_workers = int(config.get("n_workers", 1))

    raw_experiments = config.get("experiments", [])
    experiments = validate_and_load_experiments(raw_experiments)

    all_rows = []

    print(f"Starting LR Sweep across {len(experiments)} experiment setups.")
    print(f"Fixed iterations per run: {n_iterations} | Parallel workers: {n_workers}")

    for exp in experiments:
        exp_name = exp.get("name", "Unnamed_Experiment")
        env_config_path = exp["env_config"]
        env_cfg = exp["loaded_env_cfg"]
        dpp_cfg = exp["loaded_dpp_cfg"]
        param_name = exp.get("param_name", "lr")
        param_values = exp.get("param_values", [0.1])

        print(
            f"\n{'='*65}\n Experiment: {exp_name} ({dpp_cfg.get('algo', 'unknown')})\n Parameter: {param_name} over values {param_values}\n{'='*65}"
        )

        tasks = [
            (
                exp_name,
                param_name,
                val,
                env_config_path,
                copy.deepcopy(env_cfg),
                copy.deepcopy(dpp_cfg),
                n_iterations,
                mc_cfg,
            )
            for val in param_values
        ]

        exp_results = []
        if n_workers > 1 and len(tasks) > 1:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {pool.submit(_worker_run_lr, task): task[2] for task in tasks}
                for fut in as_completed(futures):
                    res = fut.result()
                    exp_results.append(res)
        else:
            for task in tasks:
                res = _worker_run_lr(task)
                exp_results.append(res)

        # Sort results by parameter value for consistent ordering
        exp_results.sort(key=lambda x: x["param_value"])
        all_rows.extend(exp_results)

        # Display Summary Table
        for r in exp_results:
            print(
                f"{param_name:<10} = {r['param_value']:<8.4g} | "
                f"MC Cost: {r['mc_cost']:>8.2f} +/- {r['mc_cost_se']*1.96:<5.2f} | "
                f"Time: {r['wall_time_s']:>5.1f}s"
            )

        # Print Best LR for this experiment
        best = min(exp_results, key=lambda x: x["mc_cost"])
        print(
            f"\n--> Best {param_name} for '{exp_name}': {best['param_value']} (Cost: {best['mc_cost']:.2f})"
        )

    csv_path = os.path.join(output_dir, "lr_sweep_results.csv")
    write_csv(all_rows, csv_path)
    print(f"\nResults saved to {csv_path}")

    plot_results(all_rows, output_dir)
    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning Rate Sweep Experiment for DPP Algorithms"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config"
    )
    parser.add_argument(
        "--plot_csv",
        nargs="+",
        type=str,
        default=None,
        metavar="CSV",
        help="Re-plot existing CSV results without re-running computation",
    )
    args = parser.parse_args()
    main(args)