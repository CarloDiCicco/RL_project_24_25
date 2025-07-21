import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from train.train_utils import train_and_eval
from utils.io import make_output_dir
from utils.logging import setup_logger

def plot_learning_curve(log_path: str):
    """
    Plot the learning curve (mean reward ± std) from evaluations.npz
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

    plt.figure()
    plt.plot(timesteps, mean_r, label="Mean Episodic Reward")
    plt.fill_between(timesteps, mean_r - std_r, mean_r + std_r, alpha=0.2, label="Std Dev")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    out_png = os.path.join(log_path, "learning_curve.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Learning curve saved to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for full RL training")
    parser.add_argument("--algo",      choices=["SAC", "PPO", "TD3"], default="SAC",
                        help="RL algorithm to use")
    parser.add_argument("--env",       default="HalfCheetah-v5",
                        help="Gymnasium environment ID")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Total training timesteps")
    args = parser.parse_args()

    logger = setup_logger("train")
    logger.info(f"Starting training: algo={args.algo}, env={args.env}, timesteps={args.timesteps}")

    # Delegate all the heavy lifting to train_and_eval()
    mean_reward = train_and_eval(
        algo=args.algo,
        env_id=args.env,
        total_timesteps=args.timesteps,
        eval_freq=100,
        n_eval_episodes=2,
        device="cuda:0",
        output_base="results",
        save_model_flag=True,
    )

    logger.info(f"Training finished — mean evaluation reward: {mean_reward:.2f}")

    # Locate the most recent results folder for this run
    base_dir = f"results/{args.algo}_{args.env}"
    all_runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    latest_run = max(all_runs, key=os.path.getmtime)

