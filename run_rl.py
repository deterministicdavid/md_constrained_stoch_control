import os
import shutil
from datetime import datetime
import argparse
import yaml
import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Our environments & eval
from sde_md.envs.sde_env import SDEDiffusionEnv, SDEDiffusionEnvDiscAct
from sde_md.evaluation.sde_env_eval import (
    plot_policy_vector_field,
    plot_value_function,
    find_value_min_max_across_time,
    PolicyEvaluator,
)
from run_rl_utils import GradientMonitorCallback

# Policies
from sde_md.policies.quantizedtabularpolicy import QuantizedTabularPolicy
from sde_md.policies.quantizedtabularqpolicy import QuantizedTabularQPolicy
from sde_md.policies.quantizedtabularqandvpolicy import QuantizedTabularQandVPolicy
from sde_md.policies.quantizedtabular_q_and_v_vtgt_policy import (
    QuantizedTabularQandVandVTgtPolicy,
)

# Algorithms
from sde_md.rl_based.tabularmd import TabularMD
from sde_md.rl_based.tabularmdqcritic import TabularMDQCritic
from sde_md.rl_based.tabularmdqandvcritic import TabularMDQandVCritic
from sde_md.rl_based.tabularmd_adv_rate_critic import TabularMDAdvRateCritic
from sde_md.rl_based.tabularmd_adv_rate_critic_td_l import TabularMDAdvRateCriticTDL



# --- REGISTRIES ---
# Map string names from YAML to actual Python classes
ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
    "md": TabularMD,
    "md_qcritic": TabularMDQCritic,
    "md_qandvcritic": TabularMDQandVCritic,
    "md_adv_rate_critic": TabularMDAdvRateCritic,
    "md_adv_rate_critic_tdl": TabularMDAdvRateCriticTDL,
}

POLICY_MAP = {
    "MlpPolicy": "MlpPolicy",
    "QuantizedTabularPolicy": QuantizedTabularPolicy,
    "QuantizedTabularQPolicy": QuantizedTabularQPolicy,
    "QuantizedTabularQandVPolicy": QuantizedTabularQandVPolicy,
    "QuantizedTabularQandVandVTgtPolicy": QuantizedTabularQandVandVTgtPolicy,
}


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def make_lr_schedule(init_lr, final_scaling):
    def schedule(f: float):
        # f goes from 1.0 (start) to 0.0 (end)
        return f * init_lr + (1.0 - f) * (init_lr * final_scaling)

    return schedule


def make_md_schedule(config, env_dt):
    mult = config.get("multiplier", 1.0)
    start_f = config.get("start_dt_factor", 1.0)
    end_f = config.get("end_dt_factor", 100.0)

    def schedule(f: float):
        return mult * (
            f * (1.0 / (start_f * env_dt)) + (1.0 - f) * (1.0 / (end_f * env_dt))
        )

    return schedule


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_yaml(args.env_config)
    learn_cfg = load_yaml(args.learn_config)

    # --- 1. Environment Setup ---
    env_type = env_cfg.pop("env_type", "discrete")
    env_cls = SDEDiffusionEnv if env_type == "continuous" else SDEDiffusionEnvDiscAct
    n_envs = learn_cfg.get("n_envs", 1)

    vec_env = make_vec_env(
        env_cls,
        n_envs=n_envs,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_cfg,
    )

    env_dt = env_cfg["dt"]

    # --- 2. Schedules & Meta Params ---
    lr_cfg = learn_cfg["learning_rate"]
    learning_rate = make_lr_schedule(lr_cfg["init"], lr_cfg["final_scaling"])

    algo_name = learn_cfg["algo"]
    algo_cls = ALGO_MAP[algo_name]

    # Auto-inject environment geometry into policy kwargs
    policy_kwargs = learn_cfg["policy"].get("kwargs", {})
    policy_kwargs["radius"] = env_cfg["radius"]
    policy_kwargs["T"] = env_cfg["T"]
    # Fallback to env_dt if policy_dt isn't specified
    if "dt" not in policy_kwargs:
        policy_kwargs["dt"] = env_dt

    policy_cls = POLICY_MAP[learn_cfg["policy"]["name"]]

    # --- 3. Gather Algo Kwargs ---
    algo_kwargs = learn_cfg.get("algo_kwargs", {})
    algo_kwargs["device"] = device
    algo_kwargs["learning_rate"] = learning_rate
    if "ent_coef" in algo_kwargs:
        algo_kwargs["ent_coef"] = float(algo_kwargs["ent_coef"])

    # If using tabular MD, attach md_lambda
    if algo_name.startswith("md"):
        md_cfg = learn_cfg.get("md_lambda", {})
        if md_cfg.get("use_schedule", False):
            algo_kwargs["md_lambda"] = make_md_schedule(md_cfg, env_dt)

    # --- 4. Paths and Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs_{algo_name}"
    tb_path = os.path.join("logs_tb", f"{algo_name}_envs{n_envs}_{timestamp}")

    if os.path.exists(tb_path):
        shutil.rmtree(tb_path)
    os.makedirs(tb_path, exist_ok=True)

    algo_kwargs["tensorboard_log"] = log_dir

    # --- 5. Build Model ---
    model = algo_cls(
        policy=policy_cls,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        env=vec_env,
        verbose=1,
        **algo_kwargs,
    )

    new_logger = configure(tb_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    monitor_callback = GradientMonitorCallback() if algo_name.startswith("md") else None

    # --- 6. Train ---
    print(f"Starting {algo_name} on {n_envs} parallel environments...")
    model.learn(total_timesteps=learn_cfg["total_timesteps"], callback=monitor_callback)

    save_file = (
        args.model if args.model else os.path.join(log_dir, f"{algo_name}_model.zip")
    )
    model.save(save_file)
    print(f"Model saved to {save_file}")
    vec_env.close()


def evaluate(args):
    if not args.model:
        raise ValueError("Must specify --model when running --evaluate.")

    env_cfg = load_yaml(args.env_config)
    learn_cfg = load_yaml(args.learn_config)

    env_type = env_cfg.pop("env_type", "discrete")
    env_cls = SDEDiffusionEnv if env_type == "continuous" else SDEDiffusionEnvDiscAct
    eval_env = env_cls(**env_cfg)

    algo_name = learn_cfg["algo"]
    algo_cls = ALGO_MAP[algo_name]

    print(f"Loading {algo_name} model from {args.model}...")
    model = algo_cls.load(args.model, env=eval_env)

    evaluator = PolicyEvaluator(model, eval_env, env_cfg)

    base_name = os.path.splitext(os.path.basename(args.model))[0]
    plot_dir = f"logs_{algo_name}"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{base_name}_eval_traj.pdf")

    n_envs = learn_cfg.get("n_envs", 4)
    evaluator.visualize_episode(deterministic=True, save_path=plot_path, N_envs=n_envs)

    radius = env_cfg["radius"]
    T = env_cfg["T"]
    dt = env_cfg["dt"]

    N_time_snapshots = 5
    vmin, vmax = find_value_min_max_across_time(
        N_time_snapshots=N_time_snapshots, radius=radius, T=T, dt=dt, model=model
    )
    for i in range(0, N_time_snapshots):
        t = max(i * (T / (N_time_snapshots - 1)) - dt, 0)
        plot_value_function(
            model, t, (-radius, radius), radius, plot_dir, 100, vmin=vmin, vmax=vmax
        )
        plot_policy_vector_field(model, eval_env, t, (-radius, radius), plot_dir, 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--env_config", type=str, default="env_config.yaml", help="Path to env YAML"
    )
    parser.add_argument(
        "--learn_config",
        type=str,
        default="learn_config.yaml",
        help="Path to learn YAML",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to save/load model"
    )

    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.evaluate:
        evaluate(args)
    else:
        parser.print_help()
