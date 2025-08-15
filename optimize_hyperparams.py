import multiprocessing as mp
import os
import json
import datetime
import numpy as np
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
)
from train.train_utils import train_and_eval

# ---------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------
ALGO          = "SAC"
ENV_ID        = "HalfCheetah-v5"
BASE_DIR      = "optuna_results"
N_TRIALS      = 10
STEPS_PER_ENV = 30_000
HYPER_RANGES  = {
    "learning_rate": [1e-3, 2.5e-3], #[1e-5, 1e-3]
    "batch_size":    [128, 128],
    "gamma":         [0.992712289037351, 0.992712289037351],
}
N_ENVS = min(mp.cpu_count(), 1)# there are 20 CPU cores on my machine (Obelisk)
TOTAL_TIMESTEPS = N_ENVS * STEPS_PER_ENV

# ---------------------------------------------------------------
# 2. Optuna objective function
# ---------------------------------------------------------------
def objective(trial: optuna.trial.Trial) -> float:
    """Single Optuna trial → returns mean evaluation reward."""
    lr    = trial.suggest_float(
        "learning_rate",
        *HYPER_RANGES["learning_rate"],
        log=True
    )
    batch = trial.suggest_int(
        "batch_size",
        HYPER_RANGES["batch_size"][0],
        HYPER_RANGES["batch_size"][1],
        step=32
    )
    gamma = trial.suggest_float("gamma", *HYPER_RANGES["gamma"])

    # define your evaluation settings
    timesteps_per_eval = TOTAL_TIMESTEPS // 2
    eval_freq_calls = timesteps_per_eval // N_ENVS
    eval_freq = eval_freq_calls
    n_eval_episodes = 2

    # call train_and_eval passing the new params
    return train_and_eval(
        algo=ALGO,
        env_id=ENV_ID,
        total_timesteps=TOTAL_TIMESTEPS,
        learning_rate=lr,
        batch_size=batch,
        gamma=gamma,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        buffer_size=1_000_000,
        output_base=BASE_DIR,
        save_model_flag=False,
    )

# ---------------------------------------------------------------
# 3. Main function
# ---------------------------------------------------------------
def main() -> None:
    print(
        f"Tuning setup: {N_ENVS} envs × {STEPS_PER_ENV} steps = "
        f"{TOTAL_TIMESTEPS:,} timesteps per trial\n"
    )

    # Create timestamped folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    PLOTS_DIR = os.path.join(
        BASE_DIR, "plots", f"{ALGO}_{ENV_ID}", timestamp
    )
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=2, gc_after_trial=True) # n_jobs=2 for parallel trials, GC to free memory

    # Save metadata
    study.trials_dataframe().to_csv(
        os.path.join(PLOTS_DIR, "trials.csv"),
        index=False
    )
    with open(os.path.join(PLOTS_DIR, "results.json"), "w") as f:
        json.dump({
            "algo":         ALGO,
            "env_id":       ENV_ID,
            "timestamp":    timestamp,
            "n_trials":     len(study.trials),
            "hyper_ranges": HYPER_RANGES,
            "best_params":  study.best_params
        }, f, indent=4)

    # Basic plots
    plot_optimization_history(study).write_image(
        os.path.join(PLOTS_DIR, "history.png")
    )
    plot_param_importances(study).write_image(
        os.path.join(PLOTS_DIR, "importances.png")
    )
    plot_parallel_coordinate(study).write_image(
        os.path.join(PLOTS_DIR, "parallel.png")
    )

    # ---------------------------------------------------------------
    # 4. Slice plot with individual x-axis settings
    # ---------------------------------------------------------------
    slice_fig = plot_slice(study)

    # 4.1 batch_size axis (column 1)
    slice_fig.update_xaxes(
        row=1,
        col=1,
        nticks=5,             # at most 5 ticks
        ticklabelstep=1,      # show every tick Plotly chooses
        tickangle=45,
        tickformat=".1e",  # scientific with 1 decimal
        automargin=True       # expand margins if labels overflow
    )

    # 4.2 gamma axis (column 2) with scientific notation
    slice_fig.update_xaxes(
        row=1,
        col=2,
        nticks=5,
        ticklabelstep=1,
        tickangle=45,
        tickformat=".1e",     # scientific with 1 decimal
        automargin=True
    )

    # 4.3 learning_rate axis (column 3) log scale + scientific
    slice_fig.update_xaxes(
        row=1,
        col=3,
        nticks=5,
        ticklabelstep=3,
        tickangle=45,
        type="log",           # log scale for learning rate
        tickformat=".1e",
        automargin=True
    )

    # Widen layout and use white template
    slice_fig.update_layout(
        width    = 1100,
        height   = 450,
        margin   = dict(l=40, r=40, t=40, b=100),
        template = "plotly_white"
    )
    slice_fig.write_image(os.path.join(PLOTS_DIR, "slice.png"))

    print(f"✓ All Optuna artefacts saved in “{PLOTS_DIR}”")

# ---------------------------------------------------------------
if __name__ == "__main__" and mp.current_process().name == "MainProcess":
    mp.freeze_support()
    main()
