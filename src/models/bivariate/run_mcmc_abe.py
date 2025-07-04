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

# Import custom modules for data processing
from src.models.utils.elog2cbs2param import elog2cbs

# Import custom modules for model estimation and prediction
from src.models.bivariate.mcmc import (
    mcmc_draw_parameters,
    draw_future_transactions
)
# ------------------------------------------------------------------
# %% 2. Load dataset and convert to CBS format
# -- 2. Load dataset and convert to CBS format
# ------------------------------------------------------------------
# We use dataset available in the BTYD package in R
# -----------------------------------------------------------------
data_path = os.path.join(project_root, "data", "processed", "cdnowElog.csv")
cdnowElog = pd.read_csv(data_path)

# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])

# Convert event log to customer-by-sufficient-statistic (CBS) format
cbs = elog2cbs(cdnowElog, units="W", T_cal="1997-09-30", T_tot="1998-06-30")
#cbs = create_customer_summary(cdnowElog, T_cal="1997-09-30", T_tot="1998-06-30")
cbs = cbs.rename(columns={"t.x": "t_x", "T.cal": "T_cal", "x.star": "x_star"})
cbs.head()
# ------------------------------------------------------------------
# %% 3. Running MCMC for model estimation M1 and M2
# -- 3. Running MCMC for model estimation M1 and M2
# ------------------------------------------------------------------
# Estimate Model M1 (no covariates)
# Track runtime for Model M1
start_m1 = time.time()
draws_m1 = mcmc_draw_parameters(
    cal_cbs =   cbs,
    covariates  =[],
    mcmc    =   4000,      # 4000 total iterations
    burnin  =   10000,   # discard the first 10 000 as warm-up
    thin    =   1,         # keep every draw; 4 000 draws after burn-in
    chains  =   2,
    seed    =   42,
    trace   =   1000,
    n_mh_steps = 20,
)
m1_duration = time.time() - start_m1
print(f"Model M1 runtime: {m1_duration:.2f} seconds")
# ------------------------------------------------------------------
# Estimate Model M2 (with covariate)
# Append dollar amount of first purchase to use as covariate (like in R)
first = (
    cdnowElog.groupby("cust")["sales"].first()
    .reset_index()
    .rename(columns={"sales": "first_sales"})
)
first["first_sales"] = first["first_sales"] * 1e-3   # scale to $10^-3
cbs = pd.merge(cbs, first[["cust", "first_sales"]], on="cust", how="left")

# Normalize first_sales
mean_val = cbs["first_sales"].mean()
std_val = cbs["first_sales"].std()
cbs["first_sales_scaled"] = (cbs["first_sales"] - mean_val) / std_val

# Estimate Model M2 (with first_sales)
# Track runtime for Model M2
start_m2 = time.time()
draws_m2 = mcmc_draw_parameters(
    cal_cbs =   cbs,
    covariates =["first_sales_scaled"],  # Parameters to replicate Abe 2009
    mcmc    =   4000,     # 14 000 total iterations
    burnin  =   10000,   # discard the first 10 000 as warm-up
    thin    =   1,         # keep every draw; 4 000 draws after burn-in
    chains  =   2,
    seed    =   42,
    trace   =   1000,
    n_mh_steps = 20,
)
m2_duration = time.time() - start_m2
print(f"Model M2 runtime: {m2_duration:.2f} seconds")
# ------------------------------------------------------------------
# Saving results
# Ensure the pickles directory exists at the project root
pickles_dir = os.path.join(project_root, "outputs", "pickles")
os.makedirs(pickles_dir, exist_ok=True)

# Save Model M1 estimates
with open(os.path.join(pickles_dir, "bivariate_M1.pkl"), "wb") as f:
    pickle.dump(draws_m1, f)

# Save Model M2 estimates
with open(os.path.join(pickles_dir, "bivariate_M2.pkl"), "wb") as f:
    pickle.dump(draws_m2, f)

# Save the CBS DataFrame
with open(os.path.join(pickles_dir, "cbs_bivariate_data.pkl"), "wb") as f:
    pickle.dump(cbs, f)

# Print runtimes
runtimes = pd.DataFrame({
    "model":   ["M1",           "M2"],
    "runtime": [m1_duration,    m2_duration]
})

# Save runtimes CSV and update 
csv_path = os.path.join(project_root, "outputs", "excel", "mcmc_runtimes.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
if os.path.exists(csv_path):
    df_runs = pd.read_csv(csv_path)
else:
    df_runs = pd.DataFrame(columns=["model", "runtime"])
for model_name, runtime in [
    ("Abe_bi_M1", m1_duration),
    ("Abe_bi_M2", m2_duration)
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