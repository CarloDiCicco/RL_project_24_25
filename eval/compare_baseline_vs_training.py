
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from utils.io import make_output_dir
from utils.logging import setup_logger

def compare(training_logs: str, random_rewards: np.ndarray,  out_dir: str):
    """
    Plot mean reward vs timesteps from evaluations.npz and
    overlay a horizontal line for the random baseline.
    Saves the plot in the provided out_dir.
    """
    logger = setup_logger("compare")
    data = np.load(training_logs)
    print("DEBUG: timesteps =", data["timesteps"])
    print("DEBUG: mean_tr   =", data["results"].mean(axis=1))
    timesteps = data["timesteps"]
    results = data["results"]            # shape: (n_eval_calls, n_episodes)
    mean_tr = results.mean(axis=1)
    std_tr = results.std(axis=1)

    # Calcola media e std della random policy (array di reward)
    rnd_mean = float(np.mean(random_rewards))
    rnd_std  = float(np.std(random_rewards))

    fig, ax = plt.subplots()
    # 1) training curve + shaded std
    ax.plot(timesteps, mean_tr, label="Trained Policy")
    ax.fill_between(timesteps,
                    mean_tr - std_tr,
                    mean_tr + std_tr,
                    color="C0", alpha=0.2,
                    label="Trained ±1 Std")
    # 2) constant line for random mean + std band
    ax.axhline(rnd_mean, linestyle="--", color="C1", label="Random Mean")
    ax.fill_between(timesteps,
                    rnd_mean - rnd_std,
                    rnd_mean + rnd_std,
                    color="C1", alpha=0.2,
                    label="Random ±1 Std")

    ax.set(xlabel="Timesteps",
           ylabel="Mean Episodic Reward",
           title="Training Curve vs Random Baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison.png"))
    plt.close()
    logger.info(f"[compare] saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_logs",
                        help="Path to evaluations.npz")
    parser.add_argument("random_rewards",
                        help="Path to .npz with random rewards (array)")
    parser.add_argument("out_dir", help="Output folder")
    args = parser.parse_args()

    # Carica array di reward episodici della random policy
    rnd = np.load(args.random_rewards)["rewards"]
    compare(args.training_logs, rnd, args.out_dir)



