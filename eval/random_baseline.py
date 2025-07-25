import argparse
import matplotlib.pyplot as plt
from env.env_setup import make_env         # use our helper to get render=True
from utils.io import make_output_dir
from utils.logging import setup_logger

def evaluate_random(env_id: str, n_episodes: int):
    """
    Run n_episodes with a random policy and return the list of episode rewards.
    """
    logger = setup_logger("random_eval")
    # creare l'env con render=True
    env = make_env(env_id, seed=0, render=True)
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        # loop fino a terminated OR truncated
        while True:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        logger.info(f"Episode {i+1}/{n_episodes} reward: {ep_reward}")
    env.close()
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v5",
                        help="Environment ID for random baseline")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of random episodes to run")
    args = parser.parse_args()

    # eseguo la baseline e raccolgo i ritorni episodici
    rewards = evaluate_random(args.env, args.episodes)
    # calcolo la media dei ritorni
    random_mean = sum(rewards) / len(rewards)

    # salvo l'istogramma
    out_dir = make_output_dir("results/random_baseline")
    plt.hist(rewards)
    plt.title("Random Policy Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.savefig(f"{out_dir}/random_rewards.png")

    logger = setup_logger("random_eval")
    logger.info(f"Saved random baseline plot to {out_dir}, mean reward = {random_mean}")
