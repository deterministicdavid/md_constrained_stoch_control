"""
Multi-Algorithm Convergence Experiment Framework.

Measures policy and value function convergence against a reference baseline
across multiple steps $N$, saving results to CSV and plotting curves.

Usage:
    python run_convergence_experiment.py --config configs/convergence_config.yaml
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

import run_dpp_based
from run_dpp_based import load_yaml, setup_env_and_policy
from sde_md.envs.sde_env import SDEDiffusionEnv, SDEDiffusionEnvDiscAct
from sde_md.iterators.dpp_based_cont_act_iterations import ContinuousActionMirrorDescent
from sde_md.iterators.dpp_based_cont_act_pia import ContinuousActionPolicyIteration
from sde_md.iterators.dpp_based_cont_act_proj_grad import ContinuousActionProjectedGradient
from sde_md.iterators.dpp_based_iterations import run_mc_eval
from sde_md.iterators.dpp_based_soft_iterations import (
    ExactMirrorDescent,
    ProjectedProbabilityDescent,
    SoftPolicyIteration,
)
from sde_md.solvers.cont_act_fdm_solver import ContinuousActionsFDMValueEvaluator
from sde_md.solvers.fdm_solver import FDMValueEvaluator
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


def create_solver(env, policy, dpp_cfg, grid_bins, ent_coef, action_radius=1.0):
    algo = dpp_cfg.get("algo", "exact_md")
    if algo == "spi":
        return SoftPolicyIteration(env, policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef)
    elif algo == "exact_md":
        return ExactMirrorDescent(env, policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef)
    elif algo == "projected_pg":
        return ProjectedProbabilityDescent(env, policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef)
    elif algo == "cont_act_md":
        return ContinuousActionMirrorDescent(env=env, policy=policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef, action_radius=action_radius)
    elif algo == "cont_act_proj_pg":
        return ContinuousActionProjectedGradient(env=env, policy=policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef, action_radius=action_radius)
    elif algo == "cont_act_pi":
        return ContinuousActionPolicyIteration(env=env, policy=policy, grid_bins=grid_bins, gamma=dpp_cfg["gamma"], ent_coef=ent_coef, action_radius=action_radius)
    else:
        raise ValueError(f"Unknown DPP algorithm: {algo}")


def load_reference(exp_cfg, recompute=False):
    ref_cfg = exp_cfg["reference"]
    model_path = ref_cfg["model"]
    env_cfg = load_yaml(ref_cfg["env_config"])
    dpp_cfg = load_yaml(ref_cfg["dpp_config"])

    if recompute:
        print(f"[{exp_cfg['name']}] Recomputing reference baseline from scratch...")
        solve_args = argparse.Namespace(env_config=ref_cfg["env_config"], dpp_config=ref_cfg["dpp_config"], model=model_path)
        run_dpp_based.solve(solve_args)
        
    grid_bins, dt, ent_coef, gamma = dpp_cfg["grid_bins"], dpp_cfg["dt"], float(dpp_cfg["ent_coef"]), dpp_cfg["gamma"]
    env, ref_policy, _, T, _ = setup_env_and_policy(env_cfg, dpp_cfg)
    ref_policy.load_state_dict(torch.load(model_path, map_location=ref_policy.device, weights_only=True))
    ref_policy.eval()

    env_type = env_cfg.get("env_type", "discrete")
    if env_type == "continuous":
        action_radius = dpp_cfg.get("algo_kwargs", {}).get("action_radius", 1.0)
        fdm_ref = ContinuousActionsFDMValueEvaluator(env, ref_policy, grid_bins=grid_bins, gamma=gamma, ent_coef=ent_coef, action_radius=action_radius)
    else:
        fdm_ref = FDMValueEvaluator(env, ref_policy, grid_bins=grid_bins, gamma=gamma, ent_coef=ent_coef)

    V_ref = fdm_ref.solve()
    print(f"[{exp_cfg['name']}] Reference baseline loaded and V_ref computed: shape {V_ref.shape}")
    return env, ref_policy, dpp_cfg, V_ref


from gymnasium import spaces

def extract_policy_parameters(policy):
    keys = list(policy.tabular_params.keys())
    
    # 1. Grab the raw tensor data based on whatever key is available
    if "action_logits" in keys: 
        raw_data = policy.tabular_params["action_logits"].detach().cpu().numpy().copy()
    elif "action_table" in keys: 
        raw_data = policy.tabular_params["action_table"].detach().cpu().numpy().copy()
    elif "Z_table" in keys: 
        raw_data = policy.tabular_params["Z_table"].detach().cpu().numpy().copy()
    else:
        # Fallback: grab the only non-value table
        param_keys = [k for k in keys if k != "value_table"]
        if len(param_keys) == 1:
            raw_data = policy.tabular_params[param_keys[0]].detach().cpu().numpy().copy()
        else:
            raise ValueError(
                f"Unknown policy parameter format in tabular_params. "
                f"Expected 'action_logits' or similar. Found keys: {keys}"
            )
            
    # 2. Determine HOW to interpret the data based on the action space!
    if isinstance(policy.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
        return ("logits", raw_data)
    else:
        # If it's a Box (continuous), then these are exact continuous action vectors,
        # even if they happen to be stored under the key "action_logits"
        return ("actions", raw_data)
    

def compute_policy_metrics(cur_param_tuple, ref_param_tuple, n_x=None):
    cur_type, cur_data = cur_param_tuple
    ref_type, ref_data = ref_param_tuple

    if cur_type == "logits" and ref_type == "logits":
        cur_p, ref_p = np.exp(cur_data), np.exp(ref_data)
        diff = cur_p - ref_p
        cur_action = np.stack([np.argmax(cur_p[:, :n_x], axis=1), np.argmax(cur_p[:, n_x:], axis=1)], axis=1) if n_x else np.argmax(cur_p, axis=1)
        ref_action = np.stack([np.argmax(ref_p[:, :n_x], axis=1), np.argmax(ref_p[:, n_x:], axis=1)], axis=1) if n_x else np.argmax(ref_p, axis=1)
        mean_kl = float(np.sum(ref_p * (np.log(ref_p + 1e-30) - np.log(cur_p + 1e-30))) / ref_p.shape[0])
        return {
            "policy_linf": float(np.max(np.abs(diff))), "policy_l1": float(np.sum(np.abs(diff))), "policy_l2": float(np.sqrt(np.mean(diff**2))),
            "action_mismatch_frac": float(np.mean(np.any(cur_action != ref_action, axis=1))) if cur_action.ndim > 1 else float(np.mean(cur_action != ref_action)),
            "mean_kl_ref_cur": mean_kl,
        }
    else:
        diff = cur_data - ref_data
        norm_diffs = np.linalg.norm(diff, axis=1) if diff.ndim > 1 else np.abs(diff)
        return {
            "policy_linf": float(np.max(np.abs(diff))), "policy_l1": float(np.sum(np.abs(diff))), "policy_l2": float(np.sqrt(np.mean(diff**2))),
            "action_mismatch_frac": float(np.mean(norm_diffs > 1e-3)), "mean_kl_ref_cur": float(np.mean(norm_diffs)),
        }


def setup_vec_env_and_run_mc_eval(policy, env_cfg, mc_cfg):
    env_class = SDEDiffusionEnv if env_cfg.get("env_type", "discrete") == "continuous" else SDEDiffusionEnvDiscAct
    env_kwargs = {k: v for k, v in env_cfg.items() if k != "env_type"}
    vec_env = make_vec_env(env_class, n_envs=mc_cfg["n_envs"], env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)
    
    start_radius, theta = 3.5, np.pi / 4
    ref_return, ref_std = run_mc_eval(
        vec_env=vec_env, policy=policy, t0=0.0, x0=np.array([start_radius * np.cos(theta), start_radius * np.sin(theta)], dtype=np.float32),
        N_mc=mc_cfg.get("N_mc", 512), seed=mc_cfg.get("seed", 42),
    )
    vec_env.close()
    return ref_return, ref_std


def evaluate_checkpoint(n_steps, env_cfg, dpp_cfg, V_ref, ref_param_tuple, n_x, mc_cfg, do_mc):
    t_start = time.time()
    env, policy = setup_env_and_policy(env_cfg, dpp_cfg)[:2]
    solver = create_solver(env, policy, dpp_cfg, dpp_cfg["grid_bins"], float(dpp_cfg["ent_coef"]), dpp_cfg.get("algo_kwargs", {}).get("action_radius", 1.0))
    solver.iterate(n_iterations=n_steps, **copy.deepcopy(dpp_cfg.get("algo_kwargs", {})))

    metrics = compute_policy_metrics(extract_policy_parameters(policy), ref_param_tuple, n_x)
    V_cur = solver.fdm_solver.solve()
    diff_v = V_cur - V_ref
    metrics.update({
        "value_linf": float(np.max(np.abs(diff_v))), "value_l2": float(np.linalg.norm(diff_v)),
        "value_rel_l2": float(np.linalg.norm(diff_v) / np.maximum(np.linalg.norm(V_ref), 1e-12))
    })

    if do_mc:
        ret, ret_std = setup_vec_env_and_run_mc_eval(policy=policy, env_cfg=env_cfg, mc_cfg=mc_cfg)
        metrics["mc_return"] = ret
        metrics["mc_return_se"] = ret_std / math.sqrt(mc_cfg["N_mc"])

    metrics["wall_time_s"] = time.time() - t_start
    return metrics


def _worker_run(args):
    n_steps, env_cfg, dpp_cfg, V_ref, ref_param_tuple, n_x, mc_cfg, do_mc = args
    return n_steps, evaluate_checkpoint(n_steps, env_cfg, dpp_cfg, V_ref, ref_param_tuple, n_x, mc_cfg, do_mc)


def write_csv(rows, csv_path):
    fieldnames = [
        "exp_name", "algo_label", "algo", "n_steps",
        "policy_linf", "policy_l1", "policy_l2", "action_mismatch_frac", "mean_kl_ref_cur",
        "value_linf", "value_l2", "value_rel_l2",
        "mc_return", "mc_return_se", "ref_return", "return_gap", "wall_time_s",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, exp_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    algo_labels = sorted(list({r["algo_label"] for r in rows}))

    # --- 1. Policy Metrics Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for label in algo_labels:
        sub_rows = sorted([r for r in rows if r["algo_label"] == label], key=lambda x: x["n_steps"])
        xs = [r["n_steps"] for r in sub_rows]
        axes[0].plot(xs, [r["policy_linf"] for r in sub_rows], marker="o", label=label)
        axes[1].plot(xs, [r["policy_l2"] for r in sub_rows], marker="s", label=label)
        axes[2].plot(xs, [r["mean_kl_ref_cur"] for r in sub_rows], marker="^", label=label)

    axes[0].set_title(r"Policy Dist $L_\infty$")
    axes[1].set_title(r"Policy Dist $L_2$")
    axes[2].set_title(r"Mean Diff / KL")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Iterations N")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(f"[{exp_name}] Policy Convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"convergence_policy_{exp_name}.pdf"), dpi=150)
    plt.close(fig)

    # --- 2. Value Metrics Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for label in algo_labels:
        sub_rows = sorted([r for r in rows if r["algo_label"] == label], key=lambda x: x["n_steps"])
        xs = [r["n_steps"] for r in sub_rows]
        axes[0].plot(xs, [r["value_linf"] for r in sub_rows], marker="o", label=label)
        axes[1].plot(xs, [r["value_rel_l2"] for r in sub_rows], marker="s", label=label)

    axes[0].set_title(r"Value $\|V_N - V_{ref}\|_\infty$")
    axes[1].set_title(r"Relative Value $\|V_N - V_{ref}\|_2 / \|V_{ref}\|_2$")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Iterations N")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(f"[{exp_name}] Value Function Convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"convergence_value_{exp_name}.pdf"), dpi=150)
    plt.close(fig)

    # --- 3. MC Return & Gap Plot ---
    # Only generate this plot if we actually ran MC evaluations
    has_mc = any("mc_return" in r for r in rows)
    if has_mc:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        
        # Extract the reference return (it's the same for all rows in this experiment)
        ref_ret = None
        for r in rows:
            if r.get("ref_return") is not None:
                ref_ret = r["ref_return"]
                break

        all_gaps = []

        for label in algo_labels:
            sub_rows = sorted([r for r in rows if r["algo_label"] == label], key=lambda x: x["n_steps"])
            xs = [r["n_steps"] for r in sub_rows]
            
            # Left Plot: Absolute MC Return
            axes[0].plot(xs, [r["mc_return"] for r in sub_rows], marker="o", label=label)
            
            # Right Plot: Return Gap (Absolute value for log scale)
            gaps = [np.abs(r["return_gap"]) for r in sub_rows]
            axes[1].plot(xs, gaps, marker="s", label=label)
            all_gaps.extend([g for g in gaps if g > 0]) # Collect for auto-scaling

        # Add the Reference Return horizontal line
        if ref_ret is not None:
            axes[0].axhline(ref_ret, color="k", ls="--", lw=1.5, label="SPI Reference")
            axes[1].axhline(0, color="k", ls="--", lw=1.5)

        axes[0].set_title("MC Average Return")
        axes[1].set_title("Return Gap (Ref - Current)")

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("Iterations N")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

        # Log-scale formatting for the Gap plot
        axes[1].set_yscale("log")
        if all_gaps:
            axes[1].set_ylim(min(all_gaps) * 0.5, max(all_gaps) * 2.0)

        fig.suptitle(f"[{exp_name}] MC Return Convergence")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"convergence_return_{exp_name}.pdf"), dpi=150)
        plt.close(fig)
        

def run_experiment(exp_cfg, global_config, args):
    exp_name = exp_cfg["name"]
    print(f"\n{'='*70}\n Running Experiment: {exp_name}\n{'='*70}")

    output_dir = exp_cfg.get("output_dir", f"results_conv_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)

    env_cfg = load_yaml(exp_cfg["env_config"])
    
    # Handle single or multiple DPP configs gracefully
    dpp_configs = exp_cfg.get("dpp_configs", {})
    if not dpp_configs and "dpp_config" in exp_cfg:
        dpp_configs = {"Algorithm": exp_cfg["dpp_config"]}
    
    recompute_ref = not args.load_ref
    _, ref_policy, _, V_ref = load_reference(exp_cfg, recompute=recompute_ref)
    n_x = len(ref_policy.action_space.nvec) if hasattr(ref_policy.action_space, "nvec") else None
    ref_param_tuple = extract_policy_parameters(ref_policy)

    mc_cfg = exp_cfg.get("mc_eval", {})
    do_mc = not args.skip_mc
    ref_return = None
    if do_mc:
        ref_return, ref_std = setup_vec_env_and_run_mc_eval(policy=ref_policy, env_cfg=env_cfg, mc_cfg=mc_cfg)
        print(f"[{exp_name}] Reference MC Return = {ref_return:.3f} ± {ref_std:.3f}")

    n_steps_list = exp_cfg.get("n_steps_list", [1, 2, 4, 8, 16])
    n_workers = int(exp_cfg.get("n_workers", global_config.get("n_workers", 1)))

    rows = []
    
    # Loop over all algorithms to test in this experiment
    for algo_label, dpp_path in dpp_configs.items():
        print(f"\n--- Evaluating Algorithm: {algo_label} ---")
        dpp_cfg = load_yaml(dpp_path)
        
        base_args = (
            copy.deepcopy(env_cfg),
            copy.deepcopy(dpp_cfg),
            V_ref,
            ref_param_tuple,
            n_x,
            mc_cfg,
            do_mc,
        )

        results = {}
        if n_workers > 1 and len(n_steps_list) > 1:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {pool.submit(_worker_run, (n, *base_args)): n for n in n_steps_list}
                for fut in as_completed(futures):
                    n, m = fut.result()
                    results[n] = m
        else:
            for n in n_steps_list:
                n_out, m = _worker_run((n, *base_args))
                results[n_out] = m

        for n in n_steps_list:
            m = results[n]
            row = {"exp_name": exp_name, "algo_label": algo_label, "algo": dpp_cfg.get("algo"), "n_steps": n, **m}
            if do_mc and ref_return is not None:
                row["ref_return"] = ref_return
                row["return_gap"] = ref_return - m.get("mc_return", 0.0)
            rows.append(row)

            print(
                f"[{algo_label}] N={n:<4} | policy Linf {m['policy_linf']:.2e} | "
                f"value Linf {m['value_linf']:.2e} | value relL2 {m['value_rel_l2']:.2e} | {m['wall_time_s']:.1f}s"
            )

    csv_path = os.path.join(output_dir, "convergence.csv")
    write_csv(rows, csv_path)
    print(f"\nResults written to {csv_path}")

    plot_results(rows, exp_name, output_dir)
    print(f"Plots saved to {output_dir}/")

def validate_experiment_configs(experiments):
    """
    Pre-flight check: Verifies that within each experiment, all tested algorithms
    share the exact same 'dt' and 'grid_bins' as the reference baseline.
    """
    print("Running pre-flight config consistency checks...")
    for exp in experiments:
        exp_name = exp.get("name", "Unnamed_Experiment")
        
        # Load reference config to get the ground truth parameters
        ref_dpp_path = exp.get("reference", {}).get("dpp_config")
        if not ref_dpp_path or not os.path.exists(ref_dpp_path):
            raise FileNotFoundError(f"[{exp_name}] Reference dpp_config missing: {ref_dpp_path}")
        
        ref_dpp_cfg = load_yaml(ref_dpp_path)
        ref_dt = ref_dpp_cfg.get("dt")
        ref_bins = ref_dpp_cfg.get("grid_bins")

        if ref_dt is None or ref_bins is None:
            raise KeyError(f"[{exp_name}] Reference config must specify 'dt' and 'grid_bins'.")

        # Handle single or multiple DPP configs gracefully
        dpp_configs = exp.get("dpp_configs", {})
        if not dpp_configs and "dpp_config" in exp:
            dpp_configs = {"Algorithm": exp["dpp_config"]}

        # Check each algorithm against the reference
        for algo_label, dpp_path in dpp_configs.items():
            if not os.path.exists(dpp_path):
                raise FileNotFoundError(f"[{exp_name}] Algorithm config missing: {dpp_path}")
            
            algo_cfg = load_yaml(dpp_path)
            algo_dt = algo_cfg.get("dt")
            algo_bins = algo_cfg.get("grid_bins")

            if algo_dt != ref_dt:
                raise ValueError(
                    f"\n[{exp_name}] Consistency Validation Failed for '{algo_label}'!\n"
                    f"Parameter 'dt' must match the reference to ensure a fair comparison.\n"
                    f"Expected: {ref_dt} (from {ref_dpp_path})\n"
                    f"Got:      {algo_dt} (in {dpp_path})"
                )
            
            if algo_bins != ref_bins:
                raise ValueError(
                    f"\n[{exp_name}] Consistency Validation Failed for '{algo_label}'!\n"
                    f"Parameter 'grid_bins' must match the reference to ensure a fair comparison.\n"
                    f"Expected: {ref_bins} (from {ref_dpp_path})\n"
                    f"Got:      {algo_bins} (in {dpp_path})"
                )
                
    print("All experimental configurations validated successfully!\n")

def main(args):
    config = load_yaml(args.config)
    
    # Allow plotting directly from a saved CSV
    if args.plot_csv:
        for path in args.plot_csv:
            rows = []
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    for k, v in list(r.items()):
                        if v not in (None, ""):
                            try: r[k] = float(v)
                            except ValueError: pass
                    if "algo_label" not in r: r["algo_label"] = r.get("algo", "Algorithm")
                    rows.append(r)
            exp_name = rows[0].get("exp_name", "Legacy_Experiment")
            plot_results(rows, exp_name, os.path.dirname(path))
            print(f"Plotted {path}")
        return

    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments defined in configuration YAML!")

    if args.exp:
        experiments = [e for e in experiments if e.get("name") == args.exp]

    validate_experiment_configs(experiments)
            
    for exp_cfg in experiments:
        run_experiment(exp_cfg, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Algorithm Convergence Experiment Framework")
    parser.add_argument("--config", type=str, default="configs/convergence_config.yaml", help="Path to YAML config")
    parser.add_argument("--exp", type=str, default=None, help="Name of specific experiment to run")
    parser.add_argument("--skip_mc", action="store_true", help="Skip MC rollout evaluation")
    parser.add_argument("--load_ref", action="store_true", help="Load existing reference model without recomputing")
    parser.add_argument("--plot_csv", nargs="+", type=str, default=None, help="Re-plot existing CSV results without re-running computation")
    args = parser.parse_args()
    main(args)