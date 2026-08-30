"""
Standalone script to plot selected TensorBoard scalars from specific experiments.

Usage example:
    python plot_tb.py --logdir ./logs_tb \
        --experiments md_adv_rate_critic_envs64_20260830_094743 md_adv_rate_critic_envs64_20260830_100622 \
        --labels "Advantage Rate (v1)" "Advantage Rate (v2)" \
        --scalars rollout/ep_rew_mean train/q_critic_loss \
        --titles "Mean Episode Reward" "Q-Critic Loss" \
        --smooth 0.96
"""

import argparse
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth_curve(scalars, weight=0.8):
    """
    Applies Exponential Moving Average (EMA) smoothing, mimicking TensorBoard's default.
    """
    if not scalars or weight == 0.0:
        return scalars
    
    smoothed = []
    last = scalars[0]
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def get_available_tags(log_dir):
    """Peek into the event file to list available scalar tags."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    return ea.Tags().get("scalars", [])


def extract_scalar(log_dir, tag):
    """Extracts step and value arrays for a specific tag."""
    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    
    if tag not in ea.Tags().get("scalars", []):
        return None, None
        
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    return steps, vals


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard logs.")
    parser.add_argument("--logdir", type=str, default="./logs_tb", help="Base directory containing experiments.")
    parser.add_argument("--experiments", "-e", nargs="+", required=True, help="List of experiment folder names to plot.")
    parser.add_argument("--labels", "-l", nargs="+", default=None, help="Custom legend labels corresponding 1-to-1 with --experiments.")
    parser.add_argument("--scalars", "-s", nargs="+", required=True, help="List of scalar tags to plot (e.g., rollout/ep_rew_mean).")
    parser.add_argument("--titles", "-t", nargs="+", default=None, help="Custom subplot titles corresponding 1-to-1 with --scalars.")
    parser.add_argument("--smooth", type=float, default=0.0, help="Smoothing weight (0.0 to 0.99). TensorBoard default is ~0.8.")
    parser.add_argument("--output", "-o", type=str, default="tb_plots", help="Output filename prefix.")
    args = parser.parse_args()

    # --- Validate Labels Mapping ---
    if args.labels:
        if len(args.labels) != len(args.experiments):
            parser.error(
                f"Number of --labels ({len(args.labels)}) must match "
                f"number of --experiments ({len(args.experiments)})."
            )
        exp_label_map = list(zip(args.experiments, args.labels))
    else:
        exp_label_map = list(zip(args.experiments, args.experiments))

    # --- Validate Titles Mapping ---
    if args.titles:
        if len(args.titles) != len(args.scalars):
            parser.error(
                f"Number of --titles ({len(args.titles)}) must match "
                f"number of --scalars ({len(args.scalars)})."
            )
        scalar_title_map = list(zip(args.scalars, args.titles))
    else:
        scalar_title_map = list(zip(args.scalars, args.scalars))

    n_scalars = len(args.scalars)
    if n_scalars == 0:
        print("No scalars provided. Exiting.")
        return

    # Determine layout: max 2 columns, dynamic rows
    cols = min(2, n_scalars)
    rows = math.ceil(n_scalars / cols)

    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(7 * cols, 5 * rows),
        squeeze=False
    )
    axes = axes.flatten()

    for idx, (scalar, title) in enumerate(scalar_title_map):
        ax = axes[idx]
        found_data = False
        
        for exp_folder, exp_label in exp_label_map:
            exp_path = os.path.join(args.logdir, exp_folder)
            
            if not os.path.exists(exp_path):
                print(f"Warning: Experiment path does not exist: {exp_path}")
                continue
                
            steps, vals = extract_scalar(exp_path, scalar)
            
            if steps is None:
                avail = get_available_tags(exp_path)
                print(f"Warning: Tag '{scalar}' not found in {exp_folder}. Available: {avail[:5]}...")
                continue
                
            found_data = True
            
            # Plot transparent raw data if smoothing is applied
            if args.smooth > 0:
                ax.plot(steps, vals, alpha=0.25)
                
            smoothed_vals = smooth_curve(vals, weight=args.smooth)
            ax.plot(steps, smoothed_vals, lw=2, label=exp_label)

        ax.set_title(title)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Value")
        ax.grid(True, which="both", alpha=0.3)
        
        if found_data:
            ax.legend()

    # Hide any unused subplots
    for idx in range(n_scalars, len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout()
    
    pdf_path = f"{args.output}.pdf"
    png_path = f"{args.output}.png"
    
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    
    print(f"Plots saved successfully to {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()