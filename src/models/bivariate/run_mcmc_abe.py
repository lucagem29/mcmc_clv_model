# --------------------------------------------------------
# --- This script runs the MCMC to reproduce Abe 2009 ---
# --------------------------------------------------------
# %% 1. Import necessary libraries & set project root & custom modules
# -- 1. Import necessary libraries & set project root & custom modules
# ------------------------------------------------------------------
import sys
import os

# ------------------------------------------------------------------
# Find project root (folder containing "src") )
# ------------------------------------------------------------------
cwd = os.getcwd()
while not os.path.isdir(os.path.join(cwd, 'src')):
    parent = os.path.dirname(cwd)
    if parent == cwd:
        break  # Reached the root of the filesystem
    cwd = parent
project_root = cwd
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------------------------------

# Import rest of libraries
import numpy as np
import pandas as pd
import pickle
import time
from IPython.display import display

# Import custom modules for model estimation and prediction
from src.models.bivariate.mcmc import (
    mcmc_draw_parameters,
    draw_future_transactions
)
# ------------------------------------------------------------------

# %% 2. Load preprocessed CBS dataset & set directories
# -- 2. Load preprocessed CBS dataset & set directories

# We use the processed CBS created by the 2A pipeline
cbs_path = os.path.join(project_root, "data", "processed", "cdnow_abeCBS.csv")
cbs = pd.read_csv(cbs_path, dtype={"cust": str}, parse_dates=["first"])
print(f"Loaded preprocessed CBS with {len(cbs)} customers.")
display(cbs.head())

# Ensure the pickles directory exists at the project root
pickles_dir = os.path.join(project_root, "outputs", "pickles")
os.makedirs(pickles_dir, exist_ok=True)

# Set the path for saving runtimes CSV
csv_path = os.path.join(project_root, "outputs", "excel", "mcmc_runtimes.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
# ------------------------------------------------------------------

# %% 3. Running MCMC for M1 
# -- 3. Running MCMC for M1

# Track runtime for Model M1
start_m1 = time.time()
draws_m1 = mcmc_draw_parameters(
    cal_cbs =   cbs,
    covariates  =[],
    mcmc    =   4000,      # 4000 total iterations
    burnin  =   10000,   # discard the first 10 000 as warm-up
    thin    =   1,         # keep every draw; 4 000 draws after burn-in
    chains  =   4,
    seed    =   42,
    trace   =   1000,
    n_mh_steps = 20,
)
m1_duration = time.time() - start_m1
print(f"Model M1 runtime: {m1_duration:.2f} seconds")

# Save Model M1 estimates
with open(os.path.join(pickles_dir, "abe_bi_m1.pkl"), "wb") as f:
    pickle.dump(draws_m1, f)
# ------------------------------------------------------------------

# %% 4. Running MCMC for M2, matching Abe 2009 | first_sales_scaled
# -- 4. Running MCMC for M2, matching Abe 2009 | first_sales_scaled

# Track runtime for Model M2
start_m2 = time.time()
draws_m2 = mcmc_draw_parameters(
    cal_cbs =   cbs,
    covariates =["first_sales_scaled"],  # Parameters to replicate Abe 2009
    mcmc    =   4000,     # 14 000 total iterations
    burnin  =   10000,   # discard the first 10 000 as warm-up
    thin    =   1,         # keep every draw; 4 000 draws after burn-in
    chains  =   4,
    seed    =   42,
    trace   =   1000,
    n_mh_steps = 20,
)
m2_duration = time.time() - start_m2
print(f"Model M2 runtime: {m2_duration:.2f} seconds")

# Save Model M2 estimates
with open(os.path.join(pickles_dir, "abe_bi_m2.pkl"), "wb") as f:
    pickle.dump(draws_m2, f)
# ------------------------------------------------------------------

# %% 5. Save Runtimes 
# -- 5. Save Runtimes
runtimes = pd.DataFrame({
    "model":   ["M1",           "M2"],
    "runtime": [m1_duration,    m2_duration]
})

# Save runtimes CSV and update 
if os.path.exists(csv_path):
    df_runs = pd.read_csv(csv_path)
else:
    df_runs = pd.DataFrame(columns=["model", "runtime"])
for model_name, runtime in [
    ("abe_bi_M1", m1_duration),
    ("abe_bi_M2", m2_duration)
]:
    # Remove any old entry for this model
    df_runs = df_runs[df_runs.model != model_name]
    # Append the new timing
    df_runs = pd.concat(
        [df_runs, pd.DataFrame([{"model": model_name, "runtime": runtime}])],
        ignore_index=True
    )
df_runs.to_csv(csv_path, index=False)
print(f"Saved runtimes to {csv_path}")
# ------------------------------------------------------------------

