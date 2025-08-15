import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils.logging import setup_logger

def compare(training_logs: str, random_rewards: np.ndarray, out_dir: str):
    """
    Training curve vs random baseline.
    - Legend outside, top-right, but *tight* to the axes (small gap).
    - More x ticks for readability.
    - Save with bbox_inches='tight' to avoid white margins.
    """
    logger = setup_logger("compare")
    data = np.load(training_logs)
    timesteps = data["timesteps"]
    results   = data["results"]            # (n_eval_calls, n_episodes)
    mean_tr   = results.mean(axis=1)
    std_tr    = results.std(axis=1)

    rnd_mean = float(np.mean(random_rewards))
    rnd_std  = float(np.std(random_rewards))

    fig, ax = plt.subplots(figsize=(11.0, 5.6))  # wider but still compact
    ax.plot(timesteps, mean_tr, label="Trained Policy", marker="o", linewidth=1.8)
    if len(timesteps) > 0 and np.any(std_tr > 0):
        ax.fill_between(timesteps, mean_tr - std_tr, mean_tr + std_tr,
                        color="C0", alpha=0.20, label="Trained ±1 Std")

    # Random baseline band (full width)
    ax.axhline(rnd_mean, linestyle="--", color="C1", label="Random Mean")
    ax.axhspan(rnd_mean - rnd_std, rnd_mean + rnd_std, facecolor="C1", alpha=0.12, label="Random ±1 Std")

    ax.set(xlabel="Timesteps (aggregated)", ylabel="Mean Episodic Reward", title="Training Curve vs Random Baseline")

    # More x ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

    # Legend OUTSIDE top-right, anchored to the axes with a *tiny* offset
    ax.legend(loc="upper left",
              bbox_to_anchor=(1.005, 1.0),      # << closer to the axes
              borderaxespad=0.0,
              frameon=False,
              handletextpad=0.4,
              columnspacing=0.8)

    out_png = os.path.join(out_dir, "comparison.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    logger.info(f"[compare] saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_logs")
    parser.add_argument("random_rewards")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    rnd = np.load(args.random_rewards)["rewards"]
    compare(args.training_logs, rnd, args.out_dir)
