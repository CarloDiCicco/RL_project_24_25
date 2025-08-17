"""
train_utils.py

Core training utilities: environment initialization, agent setup,
segmented learning with evaluation, model saving, and compatibility
helpers for Optuna tuning.
"""

import os
import math
import multiprocessing
import numpy as np
import tempfile
import shutil
from typing import Tuple
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from env.env_setup import make_env
from utils.io import make_output_dir, save_model
from utils.logging import setup_logger

# train/train_utils.py
# --- Environment options ---
# Train on harder starts (more reset noise) to learn a robust start
TRAIN_ENV_OPTS = dict(reset_noise_scale=0.15)   # try 0.15 or 0.20; at most 0.25
# Evaluate on the standard benchmark
EVAL_ENV_OPTS  = dict(reset_noise_scale=0.10)   # HalfCheetah default

# Map string identifiers to SB3 algorithm classes
ALGOS = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
}

def _to_callback_calls(eval_freq_agg: int, eval_envs) -> int:
    """
    Convert an aggregated-timestep frequency into the number of callback calls.
    With VecEnv, each env.step() advances 'num_envs' timesteps in aggregate.
    """
    n_envs = getattr(eval_envs, "num_envs", 1)
    return max(1, int(math.ceil(eval_freq_agg / n_envs)))

def init_agent(
    algo: str,
    env_id: str,
    output_base: str = "results",
    device: str = "cpu",
    save_model_flag: bool = True,
    buffer_size: int = 100_000,
    **hyperparams
):
    """
    Initialize training/evaluation vec-envs and the agent, and create the output folder.

    Returns:
        agent: SB3 agent ready to learn
        train_envs: SubprocVecEnv (VecMonitor) for training
        eval_envs: SubprocVecEnv (VecMonitor) for EvalCallback (learning curve)
        out_dir: path to the timestamped output folder
    """
    logger = setup_logger("train_utils")

    # 1) Output directory
    if save_model_flag:
        out_dir = make_output_dir(f"{output_base}/{algo}_{env_id}")
        logger.info(f"Output directory: {out_dir}")
    else:
        out_dir = tempfile.mkdtemp(prefix=f"hpo_{algo}_{env_id}_")
        logger.info(f"HPO mode: temporary directory for eval logs is {out_dir}")

    # 2) Create vectorized training environments (parallel subprocesses)
    n_envs = min(multiprocessing.cpu_count(), 1)

    # training envs: single-process (no subprocesses)
    train_envs = DummyVecEnv([
        (lambda rank=i: make_env(env_id, seed=42 + rank, render=False, **TRAIN_ENV_OPTS))
        for i in range(n_envs)
    ])
    train_envs = VecMonitor(train_envs)

    # eval envs: single-process (no subprocesses)
    eval_envs = DummyVecEnv([
        (lambda rank=i: make_env(env_id, seed=43 + rank, render=False, **EVAL_ENV_OPTS))
        for i in range(n_envs)
    ])
    eval_envs = VecMonitor(eval_envs)

    # 4) Instantiate the RL agent with provided hyperparameters
    AgentClass = ALGOS[algo]
    agent = AgentClass(
        "MlpPolicy",
        train_envs,
        verbose=0,
        device=device,
        buffer_size=buffer_size,
        **hyperparams
    )

    return agent, train_envs, eval_envs, out_dir

def learn_segment(
    agent,
    eval_envs,
    out_dir: str,
    timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    reset_num_timesteps: bool = False,
    save_best: bool = True,
    eval_freq_is_aggregated: bool = True,
    eval_callback: EvalCallback | None = None  # optional EvalCallback instance for custom behavior
):
    """
    Train the agent for a specified number of timesteps and evaluate periodically.
    """
    # Log the training segment details
    if eval_freq_is_aggregated:
        # Convert aggregated eval_freq to callback calls
        n_envs = getattr(agent.get_env(), "num_envs", 1)
        ef = max(1, eval_freq // n_envs)
    else:
        ef = eval_freq

    # Log the training segment details
    cb = eval_callback or EvalCallback(
        eval_envs,
        best_model_save_path=out_dir if save_best else None,
        log_path=out_dir,
        eval_freq=ef,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Log the training segment details
    agent.learn(
        total_timesteps=timesteps,
        callback=cb,
        reset_num_timesteps=reset_num_timesteps
    )



def close_envs(train_envs, eval_envs):
    """Close vec envs to release resources and avoid file locks."""
    train_envs.close()
    eval_envs.close()

def save_final_model(agent, out_dir: str):
    """Save a final snapshot without relying on utils.io.save_model signature."""
    import os
    # Save directly via SB3 to results/.../final_model.zip
    agent.save(os.path.join(out_dir, "final_model.zip"))


# ---------------------------------------------------------------------
# Backward-compatible function for other scripts (e.g., Optuna)
# ---------------------------------------------------------------------
def train_and_eval(
    algo: str,
    env_id: str,
    total_timesteps: int,
    eval_freq: int = 100,
    n_eval_episodes: int = 2,
    device: str = "cpu",
    output_base: str = "results",
    save_model_flag: bool = True,
    buffer_size: int = 100_000,
    **hyperparams
) -> float:
    """
    Train an RL agent end-to-end and return its mean evaluation reward.
    Preserved for backward compatibility. IMPORTANT: Here, 'eval_freq' is assumed
    to be in *callback-call units* (legacy behavior), so no conversion happens.
    """
    logger = setup_logger("train_utils")

    agent, train_envs, eval_envs, out_dir = init_agent(
        algo=algo,
        env_id=env_id,
        output_base=output_base,
        device=device,
        save_model_flag=save_model_flag,
        buffer_size=buffer_size,
        **hyperparams
    )

    learn_segment(
        agent=agent,
        eval_envs=eval_envs,
        out_dir=out_dir,
        timesteps=total_timesteps,
        eval_freq=eval_freq,                 # already in callback-call units
        n_eval_episodes=n_eval_episodes,
        reset_num_timesteps=True,
        save_best=save_model_flag,
        eval_freq_is_aggregated=False        # <-- preserve legacy semantics for Optuna
    )

    eval_file = os.path.join(out_dir, "evaluations.npz")
    with np.load(eval_file) as data:
        mean_reward = data["results"].mean()

    if save_model_flag:
        save_final_model(agent, out_dir)

    close_envs(train_envs, eval_envs)

    logger.info(f"Training complete — mean_eval_reward={mean_reward:.2f}")
    return float(mean_reward)
