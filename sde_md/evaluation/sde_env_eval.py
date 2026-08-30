import os
from typing import Tuple, Union, Dict
import numpy as np
import torch

# import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from sde_md.envs.sde_env import SDEDiffusionEnv, SDEDiffusionEnvDiscAct
from sde_md.iterators.dpp_based_iterations import run_mc_eval


def plot_policy_vector_field(
    model,
    env,
    t: float,
    plot_range: Tuple[float, float],
    plots_dir: str,
    grid_resolution=30,
):
    """
    Plots the deterministic policy as a vector field at a specific time t.
    """
    min_val, max_val = plot_range

    # 1. Retrieve Environment Parameters
    if hasattr(env, "get_attr"):
        # Vectorized environment (e.g., SubprocVecEnv)
        radius = env.get_attr("radius")[0]
        # Check if discrete action grids exist in the subprocess
        try:
            action_grids = env.get_attr("action_grids")[0]
            is_discrete = True
        except (AttributeError, Exception):
            is_discrete = False
    else:
        # Standard unvectorized environment
        real_env = env.unwrapped if hasattr(env, "unwrapped") else env
        radius = real_env.radius
        is_discrete = hasattr(real_env, "action_grids")
        if is_discrete:
            action_grids = real_env.action_grids

    # 2. Create Grid
    x = np.linspace(min_val, max_val, grid_resolution)
    y = np.linspace(min_val, max_val, grid_resolution)
    xx, yy = np.meshgrid(x, y)

    tt = np.full_like(xx, t)
    obs_batch = np.stack([xx.flatten(), yy.flatten(), tt.flatten()], axis=1)

    # 3. Get Deterministic Actions from Model
    actions, _ = model.predict(obs_batch, deterministic=True)

    # 4. Translate Actions to Physical Vectors
    if is_discrete:
        u_x = action_grids[0][actions[:, 0]]
        u_y = action_grids[1][actions[:, 1]]
    else:
        # DDPG directly outputs the continuous physical actions
        u_x = actions[:, 0]
        u_y = actions[:, 1]

    # 5. Reshape for Quiver
    u_grid = u_x.reshape(grid_resolution, grid_resolution)
    v_grid = u_y.reshape(grid_resolution, grid_resolution)

    # 6. Plot
    plt.figure(figsize=(8, 8))
    plt.quiver(xx, yy, u_grid, v_grid, color="teal", pivot="mid", scale=None)

    circle = plt.Circle(
        (0, 0), radius, color="red", fill=False, linestyle="--", linewidth=2
    )
    plt.gca().add_patch(circle)

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.title(
        f"Learned Policy Vector Field at t={t:.2f}\n(Arrows = Deterministic Action)"
    )
    plt.xlabel("State X")
    plt.ylabel("State Y")

    save_path = f"./{plots_dir}/policy_vector_field_t_{t:.2f}.pdf"
    plt.savefig(save_path)
    print(f"Vector field saved to {save_path}")
    plt.close()


def find_value_min_max_across_time(
    N_time_snapshots: int, radius: float, T: float, dt: float, model
):
    # --- PASS 1: Find global min/max across all time slices ---
    global_vmin = float("inf")
    global_vmax = float("-inf")

    print("Pre-computing global value function bounds for consistent colorbar...")
    for i in range(0, N_time_snapshots):
        t = max(i * (T / (N_time_snapshots - 1)) - dt, 0)

        # We can reuse the inner logic of plot_value_function or just call it
        # with a dummy save_path/backend. To keep it clean, we'll just extract
        # the grid logic temporarily.
        x = np.linspace(-radius, radius, 100)
        y = np.linspace(-radius, radius, 100)
        xx, yy = np.meshgrid(x, y)
        tt = np.full_like(xx, t)
        obs_batch = np.stack([xx.flatten(), yy.flatten(), tt.flatten()], axis=1)
        obs_tensor = torch.as_tensor(
            obs_batch, dtype=torch.float32, device=model.device
        )

        with torch.no_grad():
            if hasattr(model.policy, "predict_values"):
                values = model.policy.predict_values(obs_tensor)
            else:
                actions_np, _ = model.predict(obs_batch, deterministic=True)
                actions_tensor = torch.as_tensor(
                    actions_np, dtype=torch.float32, device=model.device
                )
                q1, q2 = model.policy.critic(obs_tensor, actions_tensor)
                values = torch.min(q1, q2)

        global_vmin = min(global_vmin, values.min().item())
        global_vmax = max(global_vmax, values.max().item())

    return global_vmin, global_vmax


def plot_value_function(
    model,
    t: float,
    plot_range: Tuple[float, float],
    radius: float,
    plots_dir: str,
    grid_resolution=100,
    save_path=None,
    vmin: float = None,
    vmax: float = None,
    mc_reference: Dict = None,
    algo_name: str = "",
):
    """
    Plots the value function V(x, y, t) for a fixed time slice 't'.

    Args:
        model: The trained Stable Baselines3 PPO model.
        t: The specific time at which to evaluate the value function.
        plot_range: Tuple (min, max) for the x and y axes.
        radius: Visual reference for the boundary.
        grid_resolution: Number of points per axis.
        save_path: File path to save the plot. If None, generates a default name.
    """
    if save_path is None:
        save_path = f"./{plots_dir}/value_fn_t_{t:.2f}.pdf"

    min_val, max_val = plot_range

    # 1. Create a spatial grid of states (x, y)
    x = np.linspace(min_val, max_val, grid_resolution)
    y = np.linspace(min_val, max_val, grid_resolution)
    xx, yy = np.meshgrid(x, y)

    # 2. Create the time component
    # We create a scalar array of 't' with the same shape as the grid
    tt = np.full_like(xx, t)

    # 3. Stack into Observation Batch (N*N, 3)
    # Columns: [x, y, t]
    obs_batch = np.stack([xx.flatten(), yy.flatten(), tt.flatten()], axis=1)

    # 4. Convert to Tensor
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=model.device)

    # 5. Query the Network
    with torch.no_grad():
        if hasattr(model.policy, "predict_values"):
            # Works for PPO, TabularMD, ...
            values = model.policy.predict_values(obs_tensor)
        elif hasattr(model.policy, "critic"):
            # Fallback for standard SB3 SAC
            # First, get the deterministic actions for the grid
            actions_np, _ = model.predict(obs_batch, deterministic=True)
            actions_tensor = torch.as_tensor(
                actions_np, dtype=torch.float32, device=model.device
            )

            # Query the twin Q-networks
            q1, q2 = model.policy.critic(obs_tensor, actions_tensor)

            # V(s) is roughly the minimum of the two Q-values for the chosen action
            values = torch.min(q1, q2)
            # values = q2
        else:
            raise NotImplementedError(
                "Cannot extract value function for this model architecture."
            )

    # 6. Reshape back to grid for plotting
    value_grid = values.cpu().numpy().reshape(grid_resolution, grid_resolution)

    # 7. Plot
    plt.figure(figsize=(8, 6))

    # Use pcolormesh or imshow. Imshow is faster for dense grids.
    plt.imshow(
        value_grid,
        origin="lower",
        extent=[min_val, max_val, min_val, max_val],
        cmap="viridis",
        interpolation="nearest",  # 'nearest' shows the discrete bins clearly, 'bilinear' smooths it
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=f"Predicted Value V(s, t={t:.2f})")

    # Add the "Done" boundary
    circle = plt.Circle(
        (0, 0),
        radius=radius,
        color="red",
        fill=False,
        linestyle="--",
        linewidth=2,
        label=f"Boundary (R={radius})",
    )
    plt.gca().add_patch(circle)

    prefix = f"[{algo_name}] " if algo_name else ""
    title_str = f"{prefix}Value Function at t={t:.2f}\n(Clamped at R={radius})"

    # Only display the diff if the plotted time matches the MC start time (0.0)
    if mc_reference and abs(t - mc_reference.get("t", 0.0)) < 1e-3:
        ref_x = mc_reference["x"]
        ref_y = mc_reference["y"]
        mc_mean = mc_reference["mean"]

        # Query value exactly at reference point
        ref_obs = torch.tensor(
            [[ref_x, ref_y, t]], dtype=torch.float32, device=model.device
        )
        with torch.no_grad():
            if hasattr(model.policy, "predict_values"):
                ref_val = model.policy.predict_values(ref_obs).item()
            else:
                act_np, _ = model.predict(ref_obs.cpu().numpy(), deterministic=True)
                act_t = torch.tensor(act_np, dtype=torch.float32, device=model.device)
                q1, q2 = model.policy.critic(ref_obs, act_t)
                ref_val = torch.min(q1, q2).item()

        diff = mc_mean - ref_val
        plt.plot(
            ref_x,
            ref_y,
            "w*",
            markersize=12,
            markeredgecolor="black",
            label="MC Ref Point",
        )

        title_str += f"\nMC Ref Point ({ref_x:.2f}, {ref_y:.2f}): V={ref_val:.1f} | MC={mc_mean:.1f} | Diff={diff:.2f}"

    plt.title(title_str)

    plt.xlabel("State X")
    plt.ylabel("State Y")
    plt.legend(loc="upper right")

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()


class PolicyEvaluator:
    def __init__(self, model, env, env_cfg):
        """
        Args:
            model: Trained SB3 model (PPO/TabularMD).
            env: The specific Gym environment instance (SDEDiffusionEnv or DiscAct version).
        """
        self.model = model
        self.env = env
        self.env_cfg = env_cfg

    def setup_vec_env_and_run_mc(
        self,
        t0: Union[float, None],
        x0: Union[np.array, None],
        seed: Union[int, None],
        N_mc: int,
        n_envs: int = 1,
    ):
        print(f"Setting up {n_envs} parallel environments for MC Evaluation...")
        vec_env = None
        env_type = self.env_cfg.get("env_type", "discrete")
        if env_type == "continuous":
            vec_env = make_vec_env(
                SDEDiffusionEnv,
                n_envs=n_envs,
                env_kwargs=self.env_cfg,
                vec_env_cls=SubprocVecEnv,
            )
        elif env_type == "discrete":
            vec_env = make_vec_env(
                SDEDiffusionEnvDiscAct,
                n_envs=n_envs,
                env_kwargs=self.env_cfg,
                vec_env_cls=SubprocVecEnv,
            )
        else:
            raise ValueError(f"configured env_type must be one of discrete, continuous")

        mean, stddev = run_mc_eval(
            vec_env=vec_env,
            policy=self.model.policy,
            t0=t0,
            x0=x0,
            seed=seed,
            N_mc=N_mc,
        )
        return mean, stddev

    def run_episode(self, deterministic: bool = True, ep_idx: Union[None, int] = None):
        """
        Runs a single episode and returns the trajectory data.
        """

        theta = np.random.uniform(0, 2 * np.pi)
        if ep_idx is not None:
            theta = ep_idx * np.pi / 2 + np.pi / 4

        start_radius = 4.7
        x0 = np.array(
            [start_radius * np.cos(theta), start_radius * np.sin(theta)],
            dtype=np.float32,
        )

        obs, _ = self.env.reset(options={"x0": x0, "t0": 0.0})
        done = False

        # Storage
        traj_x = []
        traj_y = []
        traj_t = []
        traj_u = []  # Physical controls
        traj_r = []  # Rewards

        # Determine if environment is discrete (has action grids)
        is_discrete = hasattr(self.env, "action_grids") or hasattr(
            self.env.unwrapped, "action_grids"
        )

        # Unwrap if necessary to access attributes
        real_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        physical_action = None
        while not done:
            # 1. Get Action from Model
            action, _ = self.model.predict(obs, deterministic=deterministic)

            # 2. Record State (Before stepping)
            # We use real_env.state to get the pure spatial state (x, y)
            current_spatial = real_env.state
            traj_x.append(current_spatial[0])
            traj_y.append(current_spatial[1])
            traj_t.append(real_env.current_time)

            # 3. Resolve Physical Action (for plotting)
            if is_discrete:
                # Map indices -> continuous values using the env's grid
                # action is (2,) indices
                u_x = real_env.action_grids[0][action[0]]
                u_y = real_env.action_grids[1][action[1]]
                physical_action = np.array([u_x, u_y])
            else:
                physical_action = action

            traj_u.append(physical_action)

            # 4. Step Environment
            obs, reward, terminated, truncated, _ = self.env.step(action)
            traj_r.append(reward)
            done = terminated or truncated

        # we're done, record final state
        current_spatial = real_env.state
        traj_x.append(current_spatial[0])
        traj_y.append(current_spatial[1])
        traj_t.append(real_env.current_time)
        traj_u.append(np.zeros_like(physical_action))

        return {
            "x": np.array(traj_x),
            "y": np.array(traj_y),
            "t": np.array(traj_t),
            "u": np.array(traj_u),
            "r": np.array(traj_r),
            "radius": real_env.radius,
        }

    def visualize_episode(
        self,
        deterministic: bool = True,
        save_path=None,
        num_episodes: int = 4,
        seed: int = 42,
        N_mc: int = 256,
        N_envs: int = 1,
        algo_name: str = "",
    ):
        """
        Runs multiple episodes and renders the trajectories with a multicolor line
        in a grid of (num_episodes x 3) subplots. Shares Y-axis limits across columns.
        Additionally saves each episode as an individual standalone PDF.
        """
        
        # --- 0. Get MC evaluation ---
        start_radius = 3.5
        theta = np.pi / num_episodes
        t0 = 0.0
        x0 = np.array(
            [start_radius * np.cos(theta), start_radius * np.sin(theta)],
            dtype=np.float32,
        )
        mean, stddev = self.setup_vec_env_and_run_mc(
            t0=t0,
            x0=x0,
            seed=seed,
            N_mc=N_mc,
            n_envs=N_envs,
        )

        mc_ref = {"x": x0[0], "y": x0[1], "t0": t0, "mean": mean}

        # --- 1. Pre-generate all episodes to compute global bounds ---
        episodes_data = [
            self.run_episode(deterministic=deterministic, ep_idx=ep_idx)
            for ep_idx in range(num_episodes)
        ]

        env_type = self.env_cfg.get("env_type", "discrete")
        if env_type == "discrete":
            a_min = self.env_cfg.get("action_min", -1.0)
            a_max = self.env_cfg.get("action_max", 1.0)
            # Safely handle lists if the config specifies bounds per dimension
            u_min = (
                np.min(a_min) if isinstance(a_min, (list, np.ndarray)) else float(a_min)
            )
            u_max = (
                np.max(a_max) if isinstance(a_max, (list, np.ndarray)) else float(a_max)
            )
        else:
            bound = float(self.env_cfg.get("action_bound", 1.0))
            u_min = -bound
            u_max = bound

        u_pad = (u_max - u_min) * 0.05 if u_max != u_min else 0.5

        # Compute global min/max for cumulative rewards
        r_min = min(min(np.min(np.cumsum(data["r"])) for data in episodes_data), mean)
        r_max = max(max(np.max(np.cumsum(data["r"])) for data in episodes_data), mean)
        r_pad = (r_max - r_min) * 0.05 if r_max != r_min else 0.5

        # --- 2. Setup Plotting Grid ---
        fig, axes = plt.subplots(num_episodes, 3, figsize=(18, 6 * num_episodes))

        prefix = f"[{algo_name}] " if algo_name else ""
        fig.suptitle(
            f"{prefix}MC Evaluation [Start: (x={x0[0]:.2f}, y={x0[1]:.2f}), t=0.0]: "
            f"Mean Return = {mean:.2f} +/- {stddev:.2f}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # If num_episodes is 1, axes is 1D. Make it 2D for consistency
        if num_episodes == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, data in enumerate(episodes_data):
            ax_traj_big, ax_ctrl_big, ax_rew_big = axes[i]
            
            # Create a separate, temporary figure for just this episode
            fig_single, (ax_traj_s, ax_ctrl_s, ax_rew_s) = plt.subplots(1, 3, figsize=(18, 6))

            # Loop over BOTH the main grid axes and the standalone axes to draw them simultaneously
            for ax_traj, ax_ctrl, ax_rew in [(ax_traj_big, ax_ctrl_big, ax_rew_big), (ax_traj_s, ax_ctrl_s, ax_rew_s)]:
                
                # --- Trajectory Plot (Left) ---
                radius = data["radius"]
                x, y, t = data["x"], data["y"], data["t"]

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                norm = plt.Normalize(t.min(), t.max())
                
                # Note: Matplotlib artists can't be shared across axes, so we instantiate the LineCollection here
                lc = LineCollection(segments, cmap="viridis", norm=norm)
                lc.set_array(t)
                lc.set_linewidth(2)

                line = ax_traj.add_collection(lc)
                # Use ax.figure to dynamically attach colorbar to the correct parent figure
                cbar = ax_traj.figure.colorbar(line, ax=ax_traj)
                cbar.set_label("Time (t)")

                circle = plt.Circle(
                    (0, 0), radius, color="red", fill=False, linestyle="--", linewidth=2, label="Boundary"
                )
                ax_traj.add_patch(circle)

                ax_traj.plot(x[0], y[0], "go", markersize=8, label="Start")
                ax_traj.plot(x[-1], y[-1], "rx", markersize=8, label="End")

                ax_traj.set_xlim(-radius * 1.1, radius * 1.1)
                ax_traj.set_ylim(-radius * 1.1, radius * 1.1)
                ax_traj.set_aspect("equal")
                ax_traj.set_xlabel("State X")
                ax_traj.set_ylabel("State Y")
                ax_traj.set_title(f"Episode {i+1}: Particle Trajectory")
                ax_traj.grid(True, alpha=0.3)
                ax_traj.legend()

                # --- Controls Plot (Center) ---
                u_vals = data["u"]
                ax_ctrl.step(t, u_vals[:, 0], where="post", label="Control X", color="teal", linewidth=2)
                if u_vals.shape[1] > 1:
                    ax_ctrl.step(t, u_vals[:, 1], where="post", label="Control Y", color="orange", linewidth=2)

                ax_ctrl.set_ylim(u_min - u_pad, u_max + u_pad)
                ax_ctrl.set_xlabel("Time (t)")
                ax_ctrl.set_ylabel("Control Effort")
                ax_ctrl.set_title(f"Episode {i+1}: Control Inputs")
                ax_ctrl.grid(True, alpha=0.3)
                ax_ctrl.legend()

                # --- Reward Plot (Right) ---
                rewards = data["r"]
                cum_rewards = np.cumsum(rewards)

                # Plot the episode's cumulative reward
                ax_rew.plot(t[1:], cum_rewards, color="purple", linewidth=2, label="Cumulative Reward")
                
                # Plot the expected MC Mean as a horizontal reference line
                ax_rew.axhline(mean, color="gray", linestyle="--", linewidth=1.5, label=f"MC Mean")

                ax_rew.set_ylim(r_min - r_pad, r_max + r_pad)
                ax_rew.set_xlabel("Time (t)")
                ax_rew.set_ylabel("Return")
                
                # Put exact values in the title for a clean "info panel" look
                ax_rew.set_title(
                    f"Ep {i+1} Return: {cum_rewards[-1]:.2f} | MC Mean: {mean:.2f} ± {stddev:.2f}"
                )
                
                ax_rew.grid(True, alpha=0.3)
                ax_rew.legend()
                
                
            # --- Format and Save Single Episode Figure ---
            fig_single.suptitle(
                f"{prefix}Episode {i+1} [Start: (x={x0[0]:.2f}, y={x0[1]:.2f}), t=0.0]",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )
            fig_single.tight_layout(rect=[0, 0, 1, 0.96])

            if save_path:
                base, ext = os.path.splitext(save_path)
                single_save_path = f"{base}_ep{i+1}{ext}"
                fig_single.savefig(single_save_path)
            
            # Close the standalone figure to free up memory
            plt.close(fig_single)

        # --- Format and Save Main Grid Figure ---
        # Call explicit figure methods rather than pyplot to avoid active-figure mismatches
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            fig.savefig(save_path)
            print(f"Evaluation plots saved to {save_path} (and individual episodes saved separately)")
        else:
            print("Lacking savepath argument, so showing fig instead.")
            plt.show()

        return mc_ref

# --- Usage Example ---
if __name__ == "__main__":
    print("Library file, don't run direct.")
