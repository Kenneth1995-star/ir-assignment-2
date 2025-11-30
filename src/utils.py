import os
import json
import numpy as np
import pandas as pd

# Here, I am writing a helper function that ensures several directories exist.
# If the directories already exist, nothing happens. Otherwise they are created.
def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# Here, I am saving metrics (either a dict or list of dicts) to a CSV file.
# I convert everything into a pandas DataFrame because it handles formatting cleanly.
def save_metrics(metrics, out_csv):
    df = pd.DataFrame([metrics]) if isinstance(metrics, dict) else pd.DataFrame(metrics)
    df.to_csv(out_csv, index=False)

