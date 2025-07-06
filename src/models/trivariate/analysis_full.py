# %% 1. Import necessary libraries & set project root & custom modules & helper function
# -- 1. Import necessary libraries & set project root & custom modules & helper function --
# ------------------------------------------------------------------

import os
import sys
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
import matplotlib.pyplot as plt
import pickle

# ---------------------------------------------------------------------
# Helper: enforce uniform decimal display (e.g. 0.63, 2.57, …)
# ---------------------------------------------------------------------
def _fmt(df: pd.DataFrame, dec: int) -> pd.DataFrame:
    """Return a copy of *df* with all float cells formatted to *dec* decimals."""
    fmt = f"{{:.{dec}f}}".format
    return df.applymap(lambda v: fmt(v) if isinstance(v, (float, np.floating)) else v)

# For interactive table display
from IPython.display import display

# ------------------------------------------------------------------------
# %% 2. Load estimates and data
# -- 2. Load estimates and data --
# ------------------------------------------------------------------
# --- Load Pre-computed Results ---
pickles_dir = os.path.join(project_root, "outputs", "pickles")

# Load MCMC draws
with open(os.path.join(pickles_dir, "full_trivariate_M1.pkl"), "rb") as f:
    draws_3pI = pickle.load(f)
with open(os.path.join(pickles_dir, "full_trivariate_M2.pkl"), "rb") as f:
    draws_3pII = pickle.load(f)

# Load the CBS DataFrame
with open(os.path.join(pickles_dir, "cbs_full_bivariate_data.pkl"), "rb") as f:
    cbs = pickle.load(f)

data_path = os.path.join(project_root, "data", "raw", "cdnow_purchases.csv")

cdnowElog = pd.read_csv(data_path)
# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])
# ------------------------------------------------------------------

DATA_DIR = os.path.join(project_root, "data", "processed") 
FILE_CBS_PATH = os.path.join(DATA_DIR, "cdnow_cbs_full.csv")

parse_cols = ["first"] if "first" in pd.read_csv(FILE_CBS_PATH, nrows=0).columns else None
cbs_df = pd.read_csv(FILE_CBS_PATH, parse_dates=parse_cols)

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

#%% 3. Table 2 – HB RFM model-fit metrics (no covariates vs. gender + age) --
# * cbs_df        : CBS table already in memory (23570 × …)
# * cdnowElog.csv : full event-log (raw transactions)
# * draws_3pI     : intercept-only RFM–M draws   (m0  – “no cov”)
# * draws_3pII    : gender_F + age_scaled draws  (m1  – “with cov”)
# ----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Build helper arrays (weeks, masks, birth_week) – run once ----------
# -----------------------------------------------------------------------
elog_real_path = os.path.join(DATA_DIR, "cdnowElog.csv")
elog_real = pd.read_csv(elog_real_path, parse_dates=["date"])

first_date = elog_real["date"].min()
elog_real["week"] = ((elog_real["date"] - first_date)
                     // pd.Timedelta("7D")).astype(int) + 1
max_week = elog_real["week"].max()

elog_sorted = elog_real.sort_values(["cust", "week"])
elog_sorted["txn_order"] = elog_sorted.groupby("cust").cumcount()
repeat_txns = elog_sorted[elog_sorted["txn_order"] >= 1]

weekly_actual = (
    repeat_txns.groupby("week")["cust"].count()
               .reindex(range(1, max_week + 1), fill_value=0)
)

times = np.arange(1, max_week + 1)
birth_week = (
    elog_real.groupby("cust")["week"].min()
             .reindex(cbs_df["cust"])
             .to_numpy()
)

weeks_cal_mask = (times >= 1)  & (times <= 39)   # calibration window
weeks_val_mask = (times >= 40) & (times <= 78)   # validation window

# -----------------------------------------------------------------------
# 3) Metric helper ------------------------------------------------------
# -----------------------------------------------------------------------
def compute_metrics(draws: dict, label: str) -> dict[str, float]:
    """Return correlation / MSE / MAPE metrics for RFM–M draws."""
    all_d = np.concatenate(draws["level_1"], axis=0)       # (D, N, 5)
    lam   = all_d[:, :, 0].mean(axis=0)
    mu    = all_d[:, :, 1].mean(axis=0)
    z     = all_d[:, :, 3].mean(axis=0)                    # P(alive)

    t_star = 39
    xstar_pred = z * (lam / mu) * (1 - np.exp(-mu * t_star))

    corr_val = np.corrcoef(cbs_df["x_star"], xstar_pred)[0, 1]
    mse_val  = np.mean((cbs_df["x_star"] - xstar_pred) ** 2)

    calib_pred = (lam / mu) * (1 - np.exp(-mu * cbs_df["T_cal"]))
    corr_cal = np.corrcoef(cbs_df["x"], calib_pred)[0, 1]
    mse_cal  = np.mean((cbs_df["x"] - calib_pred) ** 2)

    # weekly posterior-mean increments
    inc_weekly = np.zeros_like(times, dtype=float)
    n_draws, draws_per_chain = all_d.shape[0], len(draws["level_1"][0])

    for d in range(n_draws):
        ch, idx = divmod(d, draws_per_chain)
        lam_d = draws["level_1"][ch][idx, :, 0]
        mu_d  = draws["level_1"][ch][idx, :, 1]
        eta_d = draws["level_1"][ch][idx, :, 2]

        rng = np.random.default_rng(d)
        for t_idx, t in enumerate(times):
            active = (t > birth_week) & (t <= (birth_week + eta_d))
            inc_weekly[t_idx] += rng.poisson(lam=lam_d * active).sum()

    inc_weekly /= n_draws
    weekly_arr  = weekly_actual.to_numpy()

    def mape(a, p):
        cum_a = np.cumsum(a)
        cum_p = np.cumsum(p)
        return np.abs(cum_p - cum_a).mean() / cum_a[-1] * 100

    return {
        "label":     label,
        "corr_val":  corr_val,
        "corr_cal":  corr_cal,
        "mse_val":   mse_val,
        "mse_cal":   mse_cal,
        "mape_val":  mape(weekly_arr[weeks_val_mask], inc_weekly[weeks_val_mask]),
        "mape_cal":  mape(weekly_arr[weeks_cal_mask], inc_weekly[weeks_cal_mask]),
        "mape_pool": mape(weekly_arr, inc_weekly)
    }

# -----------------------------------------------------------------------
# 4) Compute metrics for both models ------------------------------------
stats_0 = compute_metrics(draws_3pI,  "HB RFM (no cov)")
stats_1 = compute_metrics(draws_3pII, "HB RFM (+ gender & age)")

# -----------------------------------------------------------------------
# 5) Assemble two-column Table 2 ----------------------------------------
table2 = pd.DataFrame({
    stats_0["label"]: [
        stats_0["corr_val"], stats_0["corr_cal"],
        stats_0["mse_val"],  stats_0["mse_cal"],
        stats_0["mape_val"], stats_0["mape_cal"], stats_0["mape_pool"]
    ],
    stats_1["label"]: [
        stats_1["corr_val"], stats_1["corr_cal"],
        stats_1["mse_val"],  stats_1["mse_cal"],
        stats_1["mape_val"], stats_1["mape_cal"], stats_1["mape_pool"]
    ]
}, index=[
    "Correlation (Validation)", "Correlation (Calibration)",
    "MSE (Validation)",         "MSE (Calibration)",
    "MAPE (Validation)",        "MAPE (Calibration)", "MAPE (Pooled)"
]).round(2)

row_order = [
    "Disaggregate measure",
    "Correlation (Validation)", "Correlation (Calibration)", "",
    "MSE (Validation)",         "MSE (Calibration)",         "",
    "Aggregate measure", "Time-series MAPE (%)",
    "MAPE (Validation)",        "MAPE (Calibration)",        "MAPE (Pooled)"
]
table2 = table2.reindex(row_order)

# Print / display
table2_disp = table2.reset_index().rename(columns={"index": ""})
print("\nTable 2. Model-fit – HB RFM, CDNOW dataset")
display(table2_disp)

#--------------------------------------------------------------------------------
# Make sure the loaded est. data is correct / does actually differ (althought just so slightly)
stats_0 = compute_metrics(draws_3pI,  "HB RFM (no cov)")
stats_1 = compute_metrics(draws_3pII, "HB RFM (+ gender & age)")

print("To verify that the loaded estimates are correct, and do differ (slightly):")
print("Raw stats M1:", stats_0)
print("Raw stats M2:", stats_1)


# %% Figure 2 – Weekly cumulative repeat transactions
# ----------------------------------------------------
# Plots three curves:
#   • Actual cumulative repeats
#   • HB RFM “no covariates”      (draws_3pI)
#   • HB RFM “+ gender & age”     (draws_3pII)
#
# Prerequisites already in memory:
#   • times, weekly_actual, birth_week   (helper block)
#   • draws_3pI   – intercept-only RFM–M
#   • draws_3pII  – gender_F + age_scaled RFM–M
# ----------------------------------------------------
def posterior_cumulative(draws: dict, label: str) -> np.ndarray:
    """
    Return posterior-mean cumulative repeat transactions per week.
    """
    inc_weekly = np.zeros_like(times, dtype=float)

    n_chains        = len(draws["level_1"])
    draws_per_chain = len(draws["level_1"][0])
    n_total_draws   = n_chains * draws_per_chain

    for d in range(n_total_draws):
        ch, idx = divmod(d, draws_per_chain)
        lam_d = draws["level_1"][ch][idx, :, 0]
        mu_d  = draws["level_1"][ch][idx, :, 1]
        eta_d = draws["level_1"][ch][idx, :, 2]

        rng = np.random.default_rng(d)            # draw-specific seed
        for t_idx, t in enumerate(times):
            active = (t > birth_week) & (t <= birth_week + eta_d)
            inc_weekly[t_idx] += rng.poisson(lam=lam_d * active).sum()

    inc_weekly /= n_total_draws                   # posterior mean
    return np.cumsum(inc_weekly)                  # cumulative curve

# -------------------------------------------------------------------------
# 1) Compute cumulative curves for both HB models
# -------------------------------------------------------------------------
cum_rfm_noCov = posterior_cumulative(draws_3pI,  "HB RFM (no cov)")
cum_rfm_cov   = posterior_cumulative(draws_3pII, "HB RFM (+ gender & age)")

# Actual cumulative repeats
cum_actual = weekly_actual.cumsum()

# -------------------------------------------------------------------------
# 2) Plot
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(times, cum_actual,      lw=2, color="tab:blue",  label="Actual")
plt.plot(times, cum_rfm_noCov,   lw=2, linestyle="--", color="tab:orange",
         label="HB RFM (no cov)")
plt.plot(times, cum_rfm_cov,     lw=2, linestyle=":",  color="tab:green",
         label="HB RFM (+ gender & age)")

plt.axvline(x=39, color="k", linestyle="--")     # calibration / validation split
plt.xlabel("Week")
plt.ylabel("Cumulative repeat transactions")
plt.title("Figure 2 – Weekly Time-Series Tracking (HB RFM)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(project_root, "outputs", "figures","Figure2_weekly_tracking_tri.png"), dpi=300, bbox_inches='tight')
plt.show()
# %%
