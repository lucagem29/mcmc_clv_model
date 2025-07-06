# %% 1. Import necessary libraries & set project root & custom modules & dataset
# -- 1. Import necessary libraries & set project root & custom modules & dataset --
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

cbs = pd.read_csv(os.path.join(project_root, "data", "processed", "cdnow_cbs_full.csv"))
# df for covariates implementation
cbs_df = cbs

# assume `sales` is total spend in calibration 
# and `x` is # of repeat transactions (i.e. excluding the first purchase)
# so (x+1) = total # of purchases in calib window
cbs_df["log_s"] = np.log( cbs_df["sales"] / (cbs_df["x"] + 1) )

# clean up infinities / NaNs (customers with zero spend)
cbs_df["log_s"] = (
    cbs_df["log_s"]
    .replace(-np.inf, 0.0)
    .fillna(0.0)
)

# Quick sanity checks
print("Shape:", cbs_df.shape)       # rows, columns
display(cbs_df.head())              # first five rows
display(cbs_df.describe(include="all").T)  # summary stats
# ------------------------------------------------------------------

# %% 2. Running MCMC for model estimation M1
# -- 2. Running MCMC for model estimation M1
# ------------------------------------------------------------------
# Estimate Model M1 (no covariates)
# Track runtime for Model M1
start_m1 = time.time()
draws_m1 = mcmc_draw_parameters(
    cal_cbs=cbs,
    covariates=[],
    mcmc=4000,      # 4000 total iterations
    burnin=10000,   # discard the first 10 000 as warm-up
    thin=1,         # keep every draw; 4 000 draws after burn-in
    chains=2,
    seed=42,
    trace=1000,
    n_mh_steps = 20,
)
m1_duration = time.time() - start_m1
print(f"Model M1 runtime: {m1_duration:.2f} seconds")
# ------------------------------------------------------------------
# Saving result M1
# Ensure the pickles directory exists at the project root
pickles_dir = os.path.join(project_root, "outputs", "pickles")
os.makedirs(pickles_dir, exist_ok=True)

# Save Model M1 estimates
with open(os.path.join(pickles_dir, "full_bivariate_M1.pkl"), "wb") as f:
    pickle.dump(draws_m1, f)
# ------------------------------------------------------------------
# %% 3. Running MCMC for model estimation M2
# -- 3. Running MCMC for model estimation M2
# ------------------------------------------------------------------

# Merge first purchase amount into cbs_df
data_path = os.path.join(project_root, "data", "raw", "cdnow_purchases.csv")
cdnowElog = pd.read_csv(data_path)

first = (
    cdnowElog.groupby("cust")["sales"].first()
    .reset_index()
    .rename(columns={"sales": "first_sales"})
)
first["first_sales"] = first["first_sales"] * 1e-3   # scale to $10^-3
cbs_df = pd.merge(
    cbs_df,
    first[["cust", "first_sales"]],
    on="cust",
    how="left"
)

# Normalize first_sales in cbs_df
mean_val = cbs_df["first_sales"].mean()
std_val  = cbs_df["first_sales"].std()
cbs_df["first_sales_scaled"] = (
    cbs_df["first_sales"] - mean_val
) / std_val


# ------------------------------------------------------------------
# gender_F  --------------------------------------------------------------
if "gender_F" not in cbs_df.columns:
    if "gender_binary" in cbs_df.columns:          # 1 = M, 0 = F
        cbs_df["gender_F"] = 1 - cbs_df["gender_binary"]
    else:
        cbs_df["gender_F"] = (cbs_df["gender"] == "F").astype(int)

# age_scaled -------------------------------------------------------------
if "age_scaled" not in cbs_df.columns:
    cbs_df["age_scaled"] = (cbs_df["age"] - cbs_df["age"].mean()) / cbs_df["age"].std()

# ------------------------------------------------------------------
# Covariates for Model M2
covariate_cols = ["first_sales_scaled", "gender_F", "age_scaled"]
print("Data shape:", cbs_df.shape)
print("Using covariates:", covariate_cols)

# Estimate Model M2 (with first_sales)
# Track runtime for Model M2
start_m2 = time.time()
draws_m2 = mcmc_draw_parameters(
    cal_cbs =   cbs_df,
    covariates = covariate_cols,  # Parameters to replicate Abe 2009
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

# Saving result M2
# Ensure the pickles directory exists at the project root
pickles_dir = os.path.join(project_root, "outputs", "pickles")
os.makedirs(pickles_dir, exist_ok=True)

# Save 
with open(os.path.join(pickles_dir, "full_bivariate_M2.pkl"), "wb") as f:
    pickle.dump(draws_m2, f)

# Save the CBS DataFrame
with open(os.path.join(pickles_dir, "cbs_full_bivariate_data.pkl"), "wb") as f:
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
    ("full_bi_M1", m1_duration),
    ("full_bi_M2", m2_duration)
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
# %%
