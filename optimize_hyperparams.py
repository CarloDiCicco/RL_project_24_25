import multiprocessing
import os
import json
import datetime
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate
)
from train.train_utils import train_and_eval

ALGO = "SAC"
ENV_ID = "HalfCheetah-v5"
BASE_DIR = "optuna_results"
N_TRIALS = 20
STEPS_PER_ENV = 10000

HYPER_RANGES = {
    "learning_rate": [1e-5, 1e-3],
    "batch_size": [64, 128, 256],
    "gamma": [0.90, 0.999]
}

N_ENVS = multiprocessing.cpu_count()
TOTAL_TIMESTEPS = N_ENVS * STEPS_PER_ENV
print(f"Tuning setup: {N_ENVS} envs × {STEPS_PER_ENV} steps/env = {TOTAL_TIMESTEPS} total timesteps per trial\n")

def objective(trial):
    lr = trial.suggest_float("learning_rate", *HYPER_RANGES["learning_rate"], log=True)
    batch = trial.suggest_categorical("batch_size", HYPER_RANGES["batch_size"])
    gamma = trial.suggest_float("gamma", *HYPER_RANGES["gamma"])
    return train_and_eval(
        algo=ALGO,
        env_id=ENV_ID,
        total_timesteps=TOTAL_TIMESTEPS,
        learning_rate=lr,
        batch_size=batch,
        gamma=gamma,
        output_base=BASE_DIR,
        save_model_flag=False
    )

if __name__ == "__main__":
    # TIMESTAMP and plots dir are defined ONCE here
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots", f"{ALGO}_{ENV_ID}", timestamp)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    # Save all trial details in CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(PLOTS_DIR, "trials.csv"), index=False)

    # Save only summary JSON (not all trial data)
    results = {
        "algo": ALGO,
        "env_id": ENV_ID,
        "timestamp": timestamp,
        "n_trials": len(study.trials),
        "hyper_ranges": HYPER_RANGES,
        "best_params": study.best_params
    }
    with open(os.path.join(PLOTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Plots: readable axis format, limit ticks
    plot_optimization_history(study).write_image(os.path.join(PLOTS_DIR, "history.png"))

    slice_fig = plot_slice(study)
    slice_fig.update_xaxes(tickformat='.0e', nticks=5)
    slice_fig.write_image(os.path.join(PLOTS_DIR, "slice.png"))

    parallel_fig = plot_parallel_coordinate(study)
    parallel_fig.update_xaxes(tickformat='.0e', nticks=5)
    parallel_fig.write_image(os.path.join(PLOTS_DIR, "parallel.png"))

    plot_param_importances(study).write_image(os.path.join(PLOTS_DIR, "importances.png"))

    print(f"Saved Optuna results under {PLOTS_DIR}/")
