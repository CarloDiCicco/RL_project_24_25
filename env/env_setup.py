import gymnasium as gym

def make_env(name: str, seed: int = 0, render: bool = False, frame_skip: int = 5):
    """
    Create and return a Gymnasium environment.
    - name: Gymnasium environment ID, e.g., "HalfCheetah-v4"
    - seed: random seed for reproducibility
    - render: if True, enable human rendering
    """
    render_mode = "human" if render else None
    env = gym.make(name, render_mode=render_mode, frame_skip=frame_skip)
    env.reset(seed=seed)
    return env