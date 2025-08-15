# eval/random_baseline.py
import argparse
import numpy as np
from env.env_setup import make_env
from utils.io import make_output_dir
from utils.logging import setup_logger

def evaluate_random(env_id: str, n_episodes: int = None, seeds=None, render: bool = False):
    """
    Run episodes with a random policy and return the list of episode rewards.
    If 'seeds' is provided, its length determines the number of episodes and
    env.reset(seed=seed) is used to ensure reproducibility across checkpoints.
    """
    logger = setup_logger("random_eval")

    if seeds is not None:
        ep_seeds = list(seeds)
    else:
        assert n_episodes is not None and n_episodes > 0, \
            "Provide either 'seeds' or a positive 'n_episodes'."
        ep_seeds = [None] * n_episodes  # no fixed seeds

    env = make_env(env_id, seed=0, render=render)
    rewards = []

    for i, s in enumerate(ep_seeds, start=1):
        obs, _ = env.reset(seed=s)  # fixed seed per episode if provided
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            done = bool(terminated or truncated)

        rewards.append(ep_reward)
        logger.info(f"Random Episode {i}/{len(ep_seeds)} reward: {ep_reward:.2f}")

    env.close()
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v5",
                        help="Environment ID for random baseline")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of random episodes to run (ignored if --seed-list is given)")
    parser.add_argument("--seed-list", type=str, default="",
                        help="Comma-separated list of seeds to use for each episode")
    args = parser.parse_args()

    seeds = None
    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(",")]

    rewards = evaluate_random(args.env, n_episodes=args.episodes, seeds=seeds, render=False)
    random_mean = float(np.mean(rewards))

    out_dir = make_output_dir("results/random_baseline")
    logger = setup_logger("random_eval")
    logger.info(f"[random baseline] mean reward = {random_mean:.2f}, episodes = {len(rewards)}; results not plotted here (handled in run_all).")
