# Reinforcement Learning Project Prototype

This repository contains a prototype for experimenting with reinforcement learning algorithms using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). The code is organized to make it easy to train agents, evaluate them and perform hyperparameter optimisation. The current version focuses on locomotion tasks (e.g. `HalfCheetah-v5` from Gymnasium), but the environment utilities can be extended for other domains.

## Requirements

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project expects a working GPU setup for training. CPU-only execution is possible but slow.

## Repository Structure

```
RL_project_24_25/
├── env/                       # Environment creation helpers
│   └── env_setup.py           # `make_env()` wrapper around Gymnasium
├── train/                     # Training logic
│   ├── train_utils.py         # `train_and_eval()` implementing the core loop
│   ├── train.py               # Command line interface for full training runs
│   └── __init__.py
├── eval/                      # Evaluation scripts
│   ├── random_baseline.py     # Collect rewards using a random policy
│   ├── policy_eval.py         # Evaluate a trained model and plot results
│   ├── compare_baseline_vs_training.py  # Plot training curves vs random baseline
│   └── __init__.py
├── utils/                     # Utility functions
│   ├── io.py                  # Output directories and model saving
│   ├── logging.py             # Minimal logging setup
│   └── __init__.py
├── optimize_hyperparams.py    # Optuna based hyperparameter search
├── run_all.py                 # Example workflow tying everything together
├── visualize.py               # Render a trained agent
└── requirements.txt
```

## Getting Started

1. **Collect a random baseline**
   ```bash
   python -m eval.random_baseline --env HalfCheetah-v5 --episodes 10
   ```
   This generates random-policy rewards under `results/random_baseline/`.

2. **Train an agent**
   ```bash
   python -m train.train --algo SAC --env HalfCheetah-v5 --timesteps 50000
   ```
   Training logs and model files are written to `results/<ALGO>_<ENV_ID>/<timestamp>/`.

3. **Visualize or evaluate**
   ```bash
   python -m visualize
   # or
   python -m eval.policy_eval path/to/best_model.zip --env HalfCheetah-v5 --algo SAC --episodes 50 --random-rewards path/to/random_baseline_rewards.npz
   ```
   The evaluation script produces plots such as reward histograms and a boxplot comparing the trained policy to the random baseline.

4. **Hyperparameter optimisation** (optional)
   ```bash
   python optimize_hyperparams.py
   ```
   Results from Optuna trials are stored under `optuna_results/` with CSV summaries and figures.

5. **End-to-end workflow**
   ```bash
   python run_all.py
   ```
   This script executes the entire workflow: random baseline, training, evaluation, and plotting.

## Current Status

- [ ] Final results and metrics – *to be added once experiments are complete*
- [ ] Detailed environment descriptions – *to be provided*
- [ ] Experiment logs and analysis – *coming soon*

## Contributing

This project is a prototype for educational purposes. Feel free to open issues or pull requests for improvements. All code is released under the MIT license.

