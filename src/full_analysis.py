# ------------------------------------------------------------------
# --------------- this script contains -----------------------------
# --------- the analysis on the full CDNOW dataset -----------------
# ------- with both bivariate and trivariate models ----------------
# ------------------------------------------------------------------

# Here we can just copy and paste the code from the two analysis_full.py scripts and merge them

#%% 1. Import libraries & set project root
#-- 1. Import libraries & set project root --
import os
from src.utils.project_root import add_project_root_to_sys_path
project_root = add_project_root_to_sys_path()
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Helper: enforce uniform decimal display (e.g. 0.63, 2.57, …)
# ---------------------------------------------------------------------
def _fmt(df: pd.DataFrame, dec: int) -> pd.DataFrame:
    """Return a copy of *df* with all float cells formatted to *dec* decimals."""
    fmt = f"{{:.{dec}f}}".format
    return df.applymap(lambda v: fmt(v) if isinstance(v, (float, np.floating)) else v)

# For interactive table display
from IPython.display import display

# ------------------------------------------------------------------
# --------------------------------------------------------------------- 
#%% 2. load estimates
#-- 2. load estimates --

# Bivariate model estimates
pickles_dir = os.path.join(project_root, "outputs", "pickles")

# Load MCMC draws
with open(os.path.join(pickles_dir, "full_bivariate_M1.pkl"), "rb") as f:
    draws_m1 = pickle.load(f)
with open(os.path.join(pickles_dir, "full_bivariate_M2.pkl"), "rb") as f:
    draws_m2 = pickle.load(f)

# Load the CBS DataFrame
with open(os.path.join(pickles_dir, "cbs_full_bivariate_data.pkl"), "rb") as f:
    cbs = pickle.load(f)
# --------------------------------------------------------------------- 

data_path = os.path.join(project_root, "data", "raw", "cdnow_purchases.csv")
cdnowElog = pd.read_csv(data_path)
# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])
# --------------------------------------------------------------------- 

# Trivariate model estimates
# Load MCMC draws
with open(os.path.join(pickles_dir, "full_trivariate_M1.pkl"), "rb") as f:
    draws_3pI = pickle.load(f)
with open(os.path.join(pickles_dir, "full_trivariate_M2.pkl"), "rb") as f:
    draws_3pII = pickle.load(f)
# --------------------------------------------------------------------- 

# Path to save the figures:
save_figures_path = os.path.join(project_root, "outputs", "figures", "full_cdnow_both")
# e.g plt.savefig(os.path.join(save_figures_path, "NAME.png"), dpi=300, bbox_inches='tight')
# --------------------------------------------------------------------- 