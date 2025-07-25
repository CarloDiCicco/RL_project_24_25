import os
from datetime import datetime

def make_output_dir(base: str = "results"):
    """
    Create and return a directory under `base/` with a timestamp.
    Example: results/20250426_153045
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, timestamp)
    os.makedirs(path, exist_ok=True)
    return path

def save_model(model, path: str):
    """
    Save the Stable-Baselines3 model to the specified path.
    The file will be named <path>/model.zip.
    """
    model.save(os.path.join(path, "final_model"))