import multiprocessing
import os
import glob
import numpy as np
from eval.random_baseline import evaluate_random  # direct import, no parsing
from train.train import plot_learning_curve

# --- Parameters (modify here if needed) ---
ENV       = "HalfCheetah-v5"
ALGO      = "SAC"
TIMESTEPS = 2_000
EPISODES  = 5
STEPS_PER_ENV = 10000

N_ENVS = multiprocessing.cpu_count()
TIMESTEPS = STEPS_PER_ENV * N_ENVS # total timesteps across all environments

print(f"User setup: {N_ENVS} envs × {STEPS_PER_ENV} steps/env = {TIMESTEPS} total timesteps\n")

# 1) Random baseline
random_rewards = evaluate_random(ENV, 5)

# 2) Training
os.system(f"python -m train.train --algo {ALGO} --env {ENV} --timesteps {TIMESTEPS}")

# 2.1) find the most recent training directory
train_dirs = glob.glob(f"results/{ALGO}_{ENV}/*")
latest     = max(train_dirs, key=os.path.getmtime)
model_path = os.path.join(latest, "best_model")

# 2.2) plot learning curve during training
plot_learning_curve(latest)

# 3) Save random-policy rewards inside the model folder
rnd_path = os.path.join(latest, "random_baseline_rewards.npz")
np.savez(rnd_path, rewards=random_rewards)

# 4) Final evaluation + boxplot
os.system(f"python -m eval.policy_eval {model_path} --env {ENV} --algo {ALGO} --episodes {EPISODES} --random-rewards {rnd_path}")

# 5) Comparison training-curve vs random baseline (mean ± std bands)
eval_log = os.path.join(latest, "evaluations.npz")
os.system(
    f"python -m eval.compare_baseline_vs_training "
    f"{eval_log} "
    f"{rnd_path} "
    f"{latest}"
)

print("\n[✔] Workflow completed! Check results/ for all plots.")