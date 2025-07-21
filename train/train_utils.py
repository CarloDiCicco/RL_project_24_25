import os
import multiprocessing
import numpy as np
import tempfile
import shutil
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from env.env_setup import make_env
from utils.io import make_output_dir, save_model
from utils.logging import setup_logger

# Map string identifiers to SB3 algorithm classes
ALGOS = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
}

def train_and_eval(
    algo: str,
    env_id: str,
    total_timesteps: int,
    eval_freq: int = 100,
    n_eval_episodes: int = 2,
    device: str = "cpu",              # explicit device argument
    output_base: str = "results",
    save_model_flag: bool = True,
    **hyperparams                  # learning_rate, batch_size, gamma, etc.
) -> float:
    """
    Train an RL agent and return its mean evaluation reward.
    """

    # 1) Set up logger for this utility
    logger = setup_logger("train_utils")

    # 2) Determine output directory
    if save_model_flag:
        out_dir = make_output_dir(f"{output_base}/{algo}_{env_id}")
        logger.info(f"Output directory: {out_dir}")
    else:
        out_dir = tempfile.mkdtemp(prefix=f"hpo_{algo}_{env_id}_")
        logger.info(f"HPO mode: temporary directory for eval logs is {out_dir}")

    # 3) Create vectorized training environments (parallel subprocesses)
    n_envs = multiprocessing.cpu_count()
    train_envs = SubprocVecEnv([
        (lambda rank=i: make_env(env_id, seed=42 + rank, render=False))
        for i in range(n_envs)
    ])
    train_envs = VecMonitor(train_envs)

    # 4) Create vectorized evaluation environments
    eval_envs = SubprocVecEnv([
        (lambda rank=i: make_env(env_id, seed=43 + rank, render=False))
        for i in range(n_envs)
    ])
    eval_envs = VecMonitor(eval_envs)

    # 5) Instantiate the RL agent with provided hyperparameters
    AgentClass = ALGOS[algo]
    agent = AgentClass(
        "MlpPolicy",
        train_envs,
        verbose=0,
        device=device,
        **hyperparams
    )

    # 6) Set up EvalCallback for periodic evaluations
    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=out_dir if save_model_flag else None,
        log_path=out_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # 7) Train the agent
    agent.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 8) Load evaluation results in a context manager to close file automatically
    eval_file = os.path.join(out_dir, "evaluations.npz")
    with np.load(eval_file) as data:
        mean_reward = data["results"].mean()

    # 9) Close environments to release Monitor and subprocess locks
    train_envs.close()
    eval_envs.close()

    # 10) Clean up temporary directory in HPO mode, retrying once on PermissionError
    if not save_model_flag:
        try:
            shutil.rmtree(out_dir)
        except PermissionError:
            shutil.rmtree(out_dir, ignore_errors=True)

    # 11) Save final model if requested
    if save_model_flag:
        save_model(agent, out_dir)

    logger.info(f"Training complete — mean_eval_reward={mean_reward:.2f}")
    return float(mean_reward)
