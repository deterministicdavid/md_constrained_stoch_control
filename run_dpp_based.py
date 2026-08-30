import os
import yaml
import argparse
import torch

from sde_md.envs.sde_env import SDEDiffusionEnv, SDEDiffusionEnvDiscAct
from sde_md.policies.quantizedtabularpolicy import QuantizedTabularPolicy
from sde_md.policies.quantizedtabularprobpolicy import QuantizedTabularProbPolicy
from sde_md.policies.projected_lqr_policy import ProjectedLQRPolicy
from sde_md.policies.quantizedtabularzpolicy import QuantizedTabularZPolicy
from sde_md.policies.quantizedtabularcontpolicy import QuantizedTabularContActionPolicy


from sde_md.iterators.dpp_based_soft_iterations import (
    SoftPolicyIteration,
    ProjectedProbabilityDescent,
    ExactMirrorDescent,
)
from sde_md.iterators.dpp_based_cont_act_pia import ContinuousActionPolicyIteration
from sde_md.iterators.dpp_based_cont_act_iterations import ContinuousActionMirrorDescent
from sde_md.iterators.dpp_based_cont_act_proj_grad import ContinuousActionProjectedGradient

# Import evaluation and plotting tools
from sde_md.evaluation.sde_env_eval import (
    PolicyEvaluator,
    plot_policy_vector_field,
    plot_value_function,
    find_value_min_max_across_time,
)


class DummyModel:
    """Wrapper to make the bare policy look like an SB3 model to the plotters."""

    def __init__(self, policy):
        self.policy = policy
        self.device = policy.device

    def predict(self, obs, deterministic=True):
        return self.policy.predict(obs, deterministic=deterministic)


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def setup_env_and_policy(env_cfg, dpp_cfg):
    """Helper to initialize the exact environment and policy from configs."""
    # Force discrete env (FDM solver requires discrete action grid)
    env_type = env_cfg.get("env_type", "discrete")
    env = None
    if env_type == "continuous":
        env = SDEDiffusionEnv(**env_cfg)
    elif env_type == "discrete":
        env = SDEDiffusionEnvDiscAct(**env_cfg)
    else:
        raise ValueError(
            f"env_type: {env_type} unknown must be one of dicrete or continuous"
        )

    T = env_cfg["T"]
    radius = env_cfg["radius"]
    dt = dpp_cfg["dt"]
    grid_bins = dpp_cfg["grid_bins"]

    policy_kwargs = {
        "grid_bins": grid_bins,
        "radius": radius,
        "T": T,
        "dt": dt,
    }
    policy = None
    algo = dpp_cfg.get("algo", "spi")
    if algo == "spi" or algo == "exact_md":
        policy = QuantizedTabularPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,
            action_logits_to_optimizer=False,
            **policy_kwargs,
        )
    elif algo == "projected_pg":
        policy = QuantizedTabularProbPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,
            action_params_to_optimizer=False,
            **policy_kwargs,
        )

    elif algo == "cont_act_pia":
        policy_kwargs["action_radius"] = dpp_cfg["algo_kwargs"].get(
            "action_radius", 1.0
        )
        policy = QuantizedTabularContActionPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=0.0,
            **policy_kwargs,
        )

    elif algo == "cont_act_md":
        policy_kwargs["action_radius"] = dpp_cfg["algo_kwargs"].get(
            "action_radius", 1.0
        )
        policy = QuantizedTabularZPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 0.0,
            **policy_kwargs,
        )

    elif algo == "cont_act_proj_pg":
        policy_kwargs["action_radius"] = dpp_cfg["algo_kwargs"].get(
            "action_radius", 1.0
        )
        policy = QuantizedTabularContActionPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=0.0,
            **policy_kwargs,
        )

    elif algo == "projected_lqr":
        # New LQR Routing
        action_bound = env_cfg.get("action_bound", 1.0)
        ctrl_cost = env_cfg.get("control_cost_factor", 0.1)
        term_cost = env_cfg.get("space_cost_factor_terminal", 2.0)
        sigma = env_cfg.get("sigma", 0.1)
        state_dim = env_cfg.get("state_dim", 2)
        policy = ProjectedLQRPolicy(
            T=T,
            action_radius=action_bound,
            control_cost=ctrl_cost,
            terminal_cost=term_cost,
            sigma=sigma,
            state_dim=state_dim,
        )

    else:
        raise ValueError(f"Unknown DPP algo: {algo}")
    policy.eval()

    return env, policy, dt, T, radius


def solve(args):
    print("=== Solving via Exact FDM Dynamic Programming ===")

    env_cfg = load_yaml(args.env_config)
    dpp_cfg = load_yaml(args.dpp_config)

    env, policy, dt, T, radius = setup_env_and_policy(env_cfg, dpp_cfg)

    grid_bins = dpp_cfg["grid_bins"]
    ent_coef = float(dpp_cfg.get("ent_coef",0.0))
    algo = dpp_cfg.get("algo", "spi")

    # Extract MC evaluation config (with safe defaults if missing)
    mc_cfg = dpp_cfg.get("mc_eval", {})
    eval_freq = mc_cfg.get("eval_freq", 0)
    N_mc = mc_cfg.get("N_mc", 100)
    n_envs = mc_cfg.get("n_envs", 8)
    eval_seed = mc_cfg.get("eval_seed", 42)

    if algo == "spi":
        solver = SoftPolicyIteration(
            env,
            policy,
            grid_bins=grid_bins,
            gamma=dpp_cfg["gamma"],
            ent_coef=ent_coef,
            env_cfg=env_cfg,
            eval_freq=eval_freq,
            n_envs=n_envs,
        )
        algo_kwargs = {"alpha": dpp_cfg.get("algo_kwargs", {}).get("alpha", 1.0)}
    elif algo == "exact_md":
        solver = ExactMirrorDescent(
            env,
            policy,
            grid_bins=grid_bins,
            gamma=dpp_cfg["gamma"],
            ent_coef=ent_coef,
            env_cfg=env_cfg,
            eval_freq=eval_freq,
            n_envs=n_envs,
        )
        algo_kwargs = {
            "md_lambda": dpp_cfg.get("algo_kwargs", {}).get("md_lambda", 0.5)
        }
    elif algo == "projected_pg":
        solver = ProjectedProbabilityDescent(
            env,
            policy,
            grid_bins=grid_bins,
            gamma=dpp_cfg["gamma"],
            ent_coef=ent_coef,
            env_cfg=env_cfg,
            eval_freq=eval_freq,
            n_envs=n_envs,
        )
        algo_kwargs = {"pg_lr": dpp_cfg.get("algo_kwargs", {}).get("pg_lr", 0.1)}
    elif algo == "cont_act_pia":
        action_radius = dpp_cfg["algo_kwargs"].get("action_radius", 1.0)
        solver = ContinuousActionPolicyIteration(
            env=env,
            policy=policy,
            grid_bins=grid_bins,
            gamma=dpp_cfg["gamma"],
            ent_coef=ent_coef,
            env_cfg=env_cfg,
            eval_freq=eval_freq,
            n_envs=n_envs,
            action_radius=action_radius,
        )
        algo_kwargs = {
            "action_radius": dpp_cfg.get("algo_kwargs", {}).get("action_radius", 10),
        }
    elif algo == "cont_act_md":
        action_radius = dpp_cfg["algo_kwargs"].get("action_radius", 1.0)
        solver = ContinuousActionMirrorDescent(
            env=env,
            policy=policy,
            grid_bins=grid_bins,
            gamma=dpp_cfg["gamma"],
            ent_coef=ent_coef,
            env_cfg=env_cfg,
            eval_freq=eval_freq,
            n_envs=n_envs,
            action_radius=action_radius,
        )
        algo_kwargs = {
            "md_lambda": dpp_cfg.get("algo_kwargs", {}).get("md_lambda", 0.5),
            "action_radius": dpp_cfg.get("algo_kwargs", {}).get("action_radius", 10),
        }
    elif algo == "cont_act_proj_pg":
        action_radius = dpp_cfg["algo_kwargs"].get("action_radius", 1.0)
        solver = ContinuousActionProjectedGradient(
            env=env,
            policy=policy,
            grid_bins=grid_bins,
            gamma=dpp_cfg["gamma"],
            ent_coef=ent_coef,
            env_cfg=env_cfg,
            eval_freq=eval_freq,
            n_envs=n_envs,
            action_radius=action_radius,
        )
        algo_kwargs = {
            "lr": dpp_cfg.get("algo_kwargs", {}).get("lr", 0.1),
            "action_radius": dpp_cfg.get("algo_kwargs", {}).get("action_radius", 10),
        }
    elif algo == "projected_lqr":
        print(
            "\nProjected LQR is an exact analytical baseline. Nothing to solve. Just run with --evaluate instead!"
        )
        return "", env, policy
    else:
        raise ValueError(f"Unknown DPP algo: {algo}")

    # Run Iteration
    V_final = solver.iterate(
        n_iterations=dpp_cfg["n_iterations"],
        eval_seed=eval_seed,
        eval_N_mc=N_mc,
        **algo_kwargs,
    )

    # Save the PyTorch policy weights
    log_dir = f"logs_dpp_{algo}"
    os.makedirs(log_dir, exist_ok=True)

    save_file = (
        args.model if args.model else os.path.join(log_dir, f"dpp_policy_{algo}.pth")
    )
    torch.save(policy.state_dict(), save_file)
    print(f"\nDPP Solve complete! Policy weights saved to {save_file}")
    return save_file, env, policy


def evaluate(args, policy=None, env=None):
    env_cfg = load_yaml(args.env_config)
    dpp_cfg = load_yaml(args.dpp_config)
    algo = dpp_cfg.get("algo", "spi")

    if policy is None and not args.model and algo != "projected_lqr":
        raise ValueError(
            "Must specify --model when running --evaluate for non-analytical models."
        )

    print(f"=== Evaluating Policy: {algo} ===")

    if policy is None and env is None:
        env, policy_new, dt, T, radius = setup_env_and_policy(env_cfg, dpp_cfg)
    else:
        dt = policy.dt
        T = policy.T
        radius = policy.radius

    if policy is None and algo != "projected_lqr":
        print("Loading policy weights...")
        policy_new.load_state_dict(
            torch.load(args.model, map_location=policy_new.device, weights_only=True)
        )
        policy = policy_new
    elif policy is None and algo == "projected_lqr":
        policy = policy_new

    print("\nGenerating Evaluation Plots...")
    plot_dir = f"logs_dpp_{algo}"
    os.makedirs(plot_dir, exist_ok=True)

    model = DummyModel(policy)

    # 1. Visualize Trajectory
    evaluator = PolicyEvaluator(model, env, env_cfg)

    if args.model:
        base_name = os.path.splitext(os.path.basename(args.model))[0]
    else:
        # Fallback for analytical baselines where args.model is None
        base_name = f"policy_{algo}"

    plot_path = os.path.join(plot_dir, f"{base_name}_eval_traj.pdf")
    mc_ref = evaluator.visualize_episode(
        deterministic=True,
        save_path=plot_path,
        num_episodes=4,
        seed=dpp_cfg["mc_eval"]["eval_seed"],
        N_mc=dpp_cfg["mc_eval"]["N_mc"],
        N_envs=dpp_cfg["mc_eval"]["n_envs"],
        algo_name=algo,
    )

    # 2. Find global value function bounds for consistent colorbars
    N_time_snapshots = 5
    vmin, vmax = find_value_min_max_across_time(
        N_time_snapshots=N_time_snapshots, radius=radius, T=T, dt=dt, model=model
    )

    # 3. Generate time-lapsed Value Function and Vector Field plots
    for i in range(0, N_time_snapshots):
        t = max(i * (T / (N_time_snapshots - 1)) - dt, 0)

        plot_value_function(
            model=model,
            t=t,
            plot_range=(-radius, radius),
            radius=radius,
            plots_dir=plot_dir,
            grid_resolution=100,
            vmin=vmin,
            vmax=vmax,
            mc_reference=mc_ref,
            algo_name=algo,
        )

        plot_policy_vector_field(
            model=model,
            env=env,
            t=t,
            plot_range=(-radius, radius),
            plots_dir=plot_dir,
            grid_resolution=20,
        )

    print(f"\nDPP Evaluation complete! Ground truth plots saved to {plot_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Exact DPP Solving or Evaluation")

    parser.add_argument(
        "--solve", action="store_true", help="Run the DPP solver and save the policy"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Load a saved policy and generate plots"
    )
    parser.add_argument(
        "--solve_and_evaluate", action="store_true", help="Run the DPP solver, save the policy, run evaluation."
    )
    parser.add_argument(
        "--env_config", type=str, default="env_config.yaml", help="Path to env YAML"
    )
    parser.add_argument(
        "--dpp_config", type=str, default="dpp_config.yaml", help="Path to DPP YAML"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to save/load the policy weights (.pth)",
    )

    args = parser.parse_args()

    if args.solve:
        solve(args)
    elif args.evaluate:
        evaluate(args)
    elif args.solve_and_evaluate:
        save_file, env, policy = solve(args)
        args.model = save_file
        evaluate(args, policy=policy, env=env)
    else:
        print("Please specify --solve or --evaluate or --solve_and_evaluate.")
        parser.print_help()
