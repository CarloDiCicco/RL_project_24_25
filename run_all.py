# run_all.py
# Orchestrates the full RL training pipeline

import multiprocessing as mp
import json
import os
from pathlib import Path
import numpy as np
from eval.random_baseline import evaluate_random
from eval.plot_learning_curve import plot_learning_curve
from eval.compare_baseline_vs_training import compare as compare_train_vs_rand
from eval.checkpoint_summary import plot_checkpoint_means
from eval.eval_suite import eval_checkpoint
from stable_baselines3.common.callbacks import EvalCallback
from train.train_utils import (
    init_agent,
    learn_segment,
    close_envs,
    save_final_model_if_any,
)

# -------------------------------
# Configuration
# -------------------------------
ENV                 = "HalfCheetah-v5"
ALGO                = "SAC"

# Final training budget (aggregated over parallel envs)
TOTAL_TIMESTEPS     = 2_000_000

# EvalCallback frequency & episodes (learning curve; given as AGGREGATED timesteps)
EVAL_FREQ_AGG       = 50_000
N_EVAL_EPISODES     = 10

# Checkpoint fractions
CHECKPOINT_FRACS    = [0.25, 0.50, 1.00]

def _check_alignment(total, fracs, freq):
    for f in fracs:
        t = int(total * f)
        if t % freq != 0:
            print(f"[WARN] checkpoint {int(f*100)}% = {t} is NOT a multiple of EVAL_FREQ_AGG={freq}")

_check_alignment(TOTAL_TIMESTEPS, CHECKPOINT_FRACS, EVAL_FREQ_AGG)

# Reproducibility seeds to reuse across baseline and checkpoints
SEEDS               = [1337 + i for i in range(10)]  # fixed list of 10 seeds

# Path to best hyperparameters (Optuna result)
BEST_PARAMS_JSON = (
    "optuna_results/plots/"
    "SAC_HalfCheetah-v5/"
    "20250810_220833/results.json"
)

def main():
    # 1) Load best hyperparameters
    with open(BEST_PARAMS_JSON, "r") as f:
        best = json.load(f)["best_params"]
    print(f"[INFO] Using optimized hyperparameters: {best}")

    # 2) Initialize agent + envs and create a timestamped output dir
    agent, train_envs, eval_envs, out_dir = init_agent(
        algo=ALGO,
        env_id=ENV,
        output_base="results",
        device="cpu",            # CPU-only
        buffer_size=1_000_000,   # large buffer (HalfCheetah ok)
        **best
    )
    out_dir_path = Path(out_dir)
    print(f"[INFO] Output dir: {out_dir}")

    # 3) Save seeds for full reproducibility
    (out_dir_path / "seeds.json").write_text(json.dumps({"seeds": SEEDS}, indent=2))

    # 4) Random baseline ONCE (1 env, fixed seeds, no render)
    random_rewards = evaluate_random(ENV, seeds=SEEDS, render=False)
    rnd_path = out_dir_path / "random_baseline_rewards.npz"
    np.savez(rnd_path, rewards=np.array(random_rewards, dtype=np.float32))
    print(f"[INFO] Saved random baseline at: {rnd_path}")

    num_envs = getattr(eval_envs, "num_envs", 1)
    ef_calls = max(1, int(np.ceil(EVAL_FREQ_AGG / num_envs)))
    shared_eval_cb = EvalCallback(
        eval_envs,
        best_model_save_path=str(out_dir_path),
        log_path=str(out_dir_path),
        eval_freq=ef_calls,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1
    )
    # ----------------------------------------------------------------------

    # 5) Train in segments and run checkpoint evaluations
    trained_so_far = 0
    for frac in CHECKPOINT_FRACS:
        target_steps  = int(TOTAL_TIMESTEPS * frac)
        segment_steps = target_steps - trained_so_far
        assert segment_steps > 0, "Non-increasing checkpoint fraction detected."

        print(f"[INFO] Training segment to reach {int(frac*100)}% "
              f"({target_steps} total timesteps | +{segment_steps} this segment)")

        # Learn this segment; reuses the shared EvalCallback
        learn_segment(
            agent=agent,
            eval_envs=eval_envs,
            out_dir=out_dir,
            timesteps=segment_steps,
            eval_freq=EVAL_FREQ_AGG,
            n_eval_episodes=N_EVAL_EPISODES,
            reset_num_timesteps=False,       # keep the global counter, training is continuous
            save_best=True,
            eval_freq_is_aggregated=True,
            eval_callback=shared_eval_cb
        )
        trained_so_far = target_steps

        # Checkpoint evaluation (10 eps, seeds fissi, con GIF)
        stage_dir = out_dir_path / "checkpoints" / f"{int(frac*100):03d}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        eval_checkpoint(
            agent=agent,
            env_id=ENV,
            stage_dir=stage_dir,
            random_rewards_path=str(rnd_path),
            seeds=SEEDS,
            algo=ALGO,
            frac_label=f"{int(frac * 100)}%",
            make_gif=True,
            gif_fps=15,
            gif_width=640
        )

        print(f"[INFO] Checkpoint {int(frac*100)}% completed: {stage_dir}")

    # 6) Final plots (learning curve and training vs random)
    plot_learning_curve(str(out_dir_path))
    eval_log = out_dir_path / "evaluations.npz"
    compare_train_vs_rand(str(eval_log), np.load(rnd_path)["rewards"], str(out_dir_path))

    # 7) Barplot of checkpoint means (25/50/100)
    plot_checkpoint_means(str(out_dir_path))

    # 8) Save final model snapshot
    save_final_model_if_any(agent, str(out_dir_path))

    # 9) Clean up vec envs
    close_envs(train_envs, eval_envs)
    print("\n[✔] Workflow completed! Check results/ for all plots.")

if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    mp.freeze_support()
    main()
