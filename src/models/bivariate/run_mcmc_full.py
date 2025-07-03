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
from IPython.display import display

# Import custom modules for data processing
from src.models.utils.elog2cbs2param import elog2cbs

# Import custom modules for model estimation and prediction
from src.models.bivariate.mcmc import (
    mcmc_draw_parameters,
    draw_future_transactions
)

cbs = pd.read_csv(os.path.join(project_root, "data", "processed", "cdnow_cbs_full.csv"))
# ------------------------------------------------------------------

# %% 3. Running MCMC for model estimation M1
# -- 3. Running MCMC for model estimation M1
# ------------------------------------------------------------------
# Estimate Model M1 (no covariates)
draws_m1 = mcmc_draw_parameters(
    cal_cbs=cbs,
    covariates=[],
    mcmc=4000,
    burnin=10000,
    thin=50,
    chains=2,
    seed=42,
    trace=1000
)

# ------------------------------------------------------------------
# Saving result M1
# Ensure the pickles directory exists at the project root
pickles_dir = os.path.join(project_root, "outputs", "pickles")
os.makedirs(pickles_dir, exist_ok=True)

# Save Model M1 estimates
with open(os.path.join(pickles_dir, "full_bivariate_M1.pkl"), "wb") as f:
    pickle.dump(draws_m1, f)
# ------------------------------------------------------------------
# %% 4. Running MCMC for model estimation M2
# -- 4. Running MCMC for model estimation M2
# ------------------------------------------------------------------
# Append dollar amount of first purchase to use as covariate (like in R)
data_path = os.path.join(project_root, "data", "raw", "cdnow_purchases.csv")
cdnowElog = pd.read_csv(data_path)

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
draws_m2 = mcmc_draw_parameters(
    cal_cbs=cbs,
    covariates=["first_sales_scaled"],
    mcmc=4000,
    burnin=10000,
    thin=50,
    chains=2,
    seed=42,
    trace=500
)

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
# ------------------------------------------------------------------