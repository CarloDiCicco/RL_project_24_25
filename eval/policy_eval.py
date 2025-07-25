
import argparse,os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, TD3
from env.env_setup import make_env
from utils.io import make_output_dir
from utils.logging import setup_logger


ALGOS = {
     "SAC": SAC,
     "PPO": PPO,
     "TD3": TD3,
}

def evaluate(model_path: str, env_id: str, algo: str, n_episodes: int, random_rewards_path: str):
    """
    Evaluate the trained agent for n_episodes, then plot a single
    boxplot comparing its return distribution with the random baseline.
    """
    logger = setup_logger("eval")
    # --- Load env and agent
    env = make_env(env_id, seed=123, render=True)
    if algo not in ALGOS:
        logger.error(f"Unknown algorithm '{algo}'. Choose from {list(ALGOS)}.")
        return
    AgentClass = ALGOS[algo]
    # load model on GPU
    agent = AgentClass.load(model_path, env=env, device="cuda:0")

    # --- Gather trained-policy returns
    trained_rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += r
        trained_rewards.append(ep_reward)
        logger.info(f"Episode {i+1}/{n_episodes} reward: {ep_reward}")
    env.close()

    # --- Load random-policy returns
    rnd = np.load(random_rewards_path)["rewards"].tolist()
    logger.info(f"Loaded {len(rnd)} random-policy rewards from {random_rewards_path}")

    # --- Boxplot comparison
    save_dir = os.path.dirname(model_path)
    plt.figure()
    plt.boxplot(
        [trained_rewards, rnd],
        labels=["Trained Policy", "Random Policy"],
        showmeans=True
    )
    plt.ylabel("Episodic Reward")
    plt.title("Final Policy Evaluation – Reward Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "policy_vs_random_boxplot.png"))
    plt.close()
    logger.info(f"Saved boxplot to {save_dir}/policy_vs_random_boxplot.png")

    # --- Histogram: granular (more bins)
    plt.figure()
    plt.hist(trained_rewards, bins=20)  # Increase bins for more granularity
    plt.title("Distribution of Episodic Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/trained_rewards_hist.png")
    plt.close()

    # Save rewards in npz for comparison scripts
    np.savez(f"{save_dir}/trained_policy_rewards.npz", rewards=trained_rewards)
    logger.info(f"Saved rewards and plots to {save_dir}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path",
                        help="Path to saved best_model.zip")
    parser.add_argument("--env", default="HalfCheetah-v5",
                        help="Environment ID")
    parser.add_argument("--algo", choices=ALGOS.keys(), required=True,
                        help="Algorithm used to train the model")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--random-rewards", required=True,
                        help="Path to .npz file with random-policy rewards")
    args = parser.parse_args()

    evaluate(
        args.model_path,
        args.env,
        args.algo,
        args.episodes,
        args.random_rewards
    )

