import os
import glob
from env.env_setup import make_env
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

# Configuration
RESULTS_DIR = "results"
ALGO = "SAC"
ENV_ID = "HalfCheetah-v5"
EPISODES = 5

# Find the most recent model directory
def find_latest_model_dir(base_dir, algo, env_id):
    search_path = os.path.join(base_dir, f"{algo}_{env_id}", "*")
    all_dirs = glob.glob(search_path)
    if not all_dirs:
        raise FileNotFoundError(f"No directories found under {search_path}")
    latest = max(all_dirs, key=os.path.getmtime)
    return latest

if __name__ == "__main__":
    # Locate latest trained model folder
    # latest_dir = find_latest_model_dir(RESULTS_DIR, ALGO, ENV_ID)
    latest_dir = "results/SAC_HalfCheetah-v5/20250609_133227"
    model_path = os.path.join(latest_dir, "best_model.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the trained agent on CPU for visualization
    AgentClass = {"SAC": SAC, "PPO": PPO, "TD3": TD3}[ALGO]
    agent = AgentClass.load(model_path, device="cpu")  # English comment: load model on CPU

    # Create a single-process environment with rendering enabled using make_env
    # 'render=True' enables human window
    render_env = DummyVecEnv([
        lambda: make_env(ENV_ID, seed=0, render=True)
    ])

    # Run and render episodes
    for ep in range(1, EPISODES + 1):
        # Reset may return obs or (obs, info)
        reset_out = render_env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out

        done = False
        total_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            step_out = render_env.step(action)
            # Handle different VecEnv step APIs
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated[0] or truncated[0])
                reward_val = reward[0]
            elif len(step_out) == 4:
                obs, rewards, dones, _ = step_out
                done = bool(dones[0])
                reward_val = rewards[0]
            else:
                raise ValueError(f"Unexpected step output length: {len(step_out)}")

            total_reward += reward_val

        print(f"[INFO] Episode {ep} reward: {total_reward:.2f}")

    render_env.close()