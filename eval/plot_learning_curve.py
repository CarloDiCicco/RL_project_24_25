# eval/plot_learning_curve.py
import os
import numpy as np
import matplotlib.pyplot as plt

# Keep charts consistent for README: 1000x600 px
DPI     = 100
FIGSIZE = (10, 6)  # inches -> 10*100 x 6*100 = 1000x600 px

def plot_learning_curve(log_path: str):
    """
    Plot the learning curve (mean reward ± std) from evaluations.npz.
    X-axis = aggregated timesteps (model.num_timesteps at eval time).
    """
    eval_file = os.path.join(log_path, "evaluations.npz")
    if not os.path.exists(eval_file):
        print(f"No evaluation log found at {eval_file}")
        return

    data = np.load(eval_file)
    timesteps = data["timesteps"]
    results   = data["results"]  # shape: (n_calls, n_episodes)
    mean_r    = results.mean(axis=1)
    std_r     = results.std(axis=1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(timesteps, mean_r, label="Mean Episodic Reward", marker="o", linewidth=1.8)
    if len(timesteps) > 0:
        ax.fill_between(timesteps, mean_r - std_r, mean_r + std_r, alpha=0.20, label="Std Dev")

    ax.set_xlabel("Timesteps (aggregated)")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curve")
    ax.legend(loc="upper left")

    # Save at fixed size (no tight bbox → consistent pixel size)
    out_png = os.path.join(log_path, "learning_curve.png")
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"Learning curve saved to {out_png} (1000x600 px)")
