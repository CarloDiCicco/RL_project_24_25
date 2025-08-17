"""
env_setup.py

Factory for Gymnasium MuJoCo environments with configurable
frame skip and reset noise. Provides make_env() helper.
"""

import gymnasium as gym

def make_env(
    name: str,
    seed: int = 0,
    render: bool = False,
    frame_skip: int = 5,
    reset_noise_scale: float = 0.10,  # HalfCheetah-v5 default
):
    """
    Create and return a Gymnasium MuJoCo environment.
    `reset_noise_scale` controls how much the initial state is perturbed at reset.
    """
    render_mode = "human" if render else None
    env = gym.make(
        name,
        render_mode=render_mode,
        frame_skip=frame_skip,
        reset_noise_scale=reset_noise_scale,
    )
    env.reset(seed=seed)
    return env
