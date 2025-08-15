# eval/checkpoint_summary.py

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Consistent export for README: 1000x600 px
DPI     = 100
FIGSIZE = (10, 6)  # inches -> 1000x600 px

def _algo_env_from_run_dir(run_dir: Path):
    """
    Infer algo and env from parent folder name like 'SAC_HalfCheetah-v5'.
    """
    pair = run_dir.parent.name  # e.g. 'SAC_HalfCheetah-v5'
    if "_" in pair:
        algo, env = pair.split("_", 1)
    else:
        algo, env = "Agent", pair
    return algo, env

def plot_checkpoint_means(run_dir: str):
    """
    Read rewards from checkpoints/025, /050, /100 and save a barplot of
    their mean rewards as 'checkpoint_means_barplot.png' in run_dir.
    Exported at a fixed 1000x600 px to keep README visuals consistent.
    """
    rd = Path(run_dir)
    cp_dirs = [rd / "checkpoints" / "025",
               rd / "checkpoints" / "050",
               rd / "checkpoints" / "100"]
    labels = ["25%", "50%", "100%"]
    means = []

    for cp in cp_dirs:
        npz_path = cp / "trained_rewards.npz"
        if not npz_path.exists():
            means.append(np.nan)
            continue
        rewards = np.load(npz_path)["rewards"]
        means.append(float(np.mean(rewards)))

    # If all are NaN, nothing to plot
    if all(np.isnan(m) for m in means):
        print("[checkpoint_summary] No checkpoint rewards found — skipping barplot.")
        return

    algo, env = _algo_env_from_run_dir(rd)

    # Determine N from the first available checkpoint (episodes per checkpoint)
    N = next(
        (len(np.load(cp / "trained_rewards.npz")["rewards"]) for cp in cp_dirs if (cp / "trained_rewards.npz").exists()),
        None
    )
    ylabel = f"Mean episodic return (N={N} per checkpoint)" if N is not None else "Mean episodic return"

    # Use fixed figure size for consistent pixel output
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, means)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Checkpoint Mean Returns — {algo} on {env}", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)

    # Value labels on top of bars
    for bar, val in zip(bars, means):
        txt = "N/A" if np.isnan(val) else f"{val:.0f}"
        y = 0.0 if np.isnan(val) else val
        ax.text(bar.get_x() + bar.get_width()/2.0, y, txt,
                ha="center", va="bottom", fontsize=9)

    # Keep layout tidy without changing the final pixel size
    fig.tight_layout()

    out_png = rd / "checkpoint_means_barplot.png"
    fig.savefig(out_png, dpi=DPI)  # no bbox_inches='tight' → exact 1000x600 px
    plt.close(fig)
    print(f"[checkpoint_summary] Saved {out_png} (1000x600 px)")
