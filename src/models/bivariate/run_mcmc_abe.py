# %% 1. Import necessary libraries & set project root & custom modules
# -- 1. Import necessary libraries & set project root & custom modules
# ------------------------------------------------------------------
import sys
import os
from src.utils.project_root import add_project_root_to_sys_path
project_root = add_project_root_to_sys_path()
# ------------------------------------------------------------------
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
# ------------------------------------------------------------------