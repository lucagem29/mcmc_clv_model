# ------------------------------------------------------------------
# --------------- this script contains -----------------------------
# --------- the analysis on the full CDNOW dataset -----------------
# ------- with both bivariate and trivariate models ----------------
# ------------------------------------------------------------------

# -----------------------------------------------------------------

# %% 1. Import necessary libraries
# -- 1. Import necessary libraries --
#import os
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Add lifetimes ParetoNBDFitter for MLE baseline
from lifetimes import ParetoNBDFitter

# Set up the project root directory
cwd = os.getcwd()
while not os.path.isdir(os.path.join(cwd, 'src')):
    parent = os.path.dirname(cwd)
    if parent == cwd:
        break  # Reached the root of the filesystem
    cwd = parent
project_root = cwd
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------
# Helper: enforce uniform decimal display (e.g. 0.63, 2.57, …)
# ---------------------------------------------------------------------
def _fmt(df: pd.DataFrame, dec: int) -> pd.DataFrame:
    """Return a copy of *df* with all float cells formatted to *dec* decimals."""
    fmt = f"{{:.{dec}f}}".format
    return df.applymap(lambda v: fmt(v) if isinstance(v, (float, np.floating)) else v)
# ------------------------------------------------------------------

# Path to save the figures:
save_figures_path = os.path.join(project_root, "outputs", "figures", "full_cdnow_both")
# e.g plt.savefig(os.path.join(save_figures_path, "NAME.png"), dpi=300, bbox_inches='tight')

# %% 2. Load estimated parameters and data
# -- 2. Load estimated parameters and data --
# Set up the directory for pickles
pickles_dir = os.path.join(project_root, "outputs", "pickles")
#-----------
#------------------------------
#-------------------------------------------------
# CHANGE TO FULL CDNOW DATASET ONCE IT ALL RUNS; DELETE THIS COMMENT AFTERWARDS
#-------------------------------------------------
#------------------------------
# Load Estimates
# Bivariate
with open(os.path.join(pickles_dir, "filtered_bivariate_M1.pkl"), "rb") as f:
    bi_m1 = pickle.load(f)
with open(os.path.join(pickles_dir, "filtered_bivariate_M2.pkl"), "rb") as f:
    bi_m2 = pickle.load(f)

# Trivariate
with open(os.path.join(pickles_dir, "trivariate_M1.pkl"), "rb") as f:
    tri_m1 = pickle.load(f)
with open(os.path.join(pickles_dir, "trivariate_M2.pkl"), "rb") as f:
    tri_m2 = pickle.load(f)

# CBS data
with open(os.path.join(pickles_dir, "cbs_filtered_bivariate_data.pkl"), "rb") as f:
    cbs = pickle.load(f)

# -----------------
# Remark: Do we also need something like `cbs_full_trivariate_data.pkl`??
# -----------------

# Elog data (1/10 CDNOW dataset)
data_path = os.path.join(project_root, "data", "processed", "cdnowElog.csv")
cdnowElog = pd.read_csv(data_path)
# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])
# ----------------------------------------------------------------------------------------
# 
# %% 3. Descriptive statistics
# -- 3. Descriptive statistics --
table1_stats = pd.DataFrame(
    {
        "Mean": [
            cbs["x"].mean(),
            cbs["T_cal"].mean() * 7,  # weeks to days
            (cbs["T_cal"] - cbs["t_x"]).mean() * 7,  # weeks to days
            cdnowElog.groupby("cust")["sales"].first().mean()
        ],
        "Std. dev.": [
            cbs["x"].std(),
            cbs["T_cal"].std() * 7,
            (cbs["T_cal"] - cbs["t_x"]).std() * 7,
            cdnowElog.groupby("cust")["sales"].first().std()
        ],
        "Min": [
            cbs["x"].min(),
            cbs["T_cal"].min() * 7,
            (cbs["T_cal"] - cbs["t_x"]).min() * 7,
            cdnowElog.groupby("cust")["sales"].first().min()
        ],
        "Max": [
            cbs["x"].max(),
            cbs["T_cal"].max() * 7,
            (cbs["T_cal"] - cbs["t_x"]).max() * 7,
            cdnowElog.groupby("cust")["sales"].first().max()
        ],
    },
    index=[
        "Number of repeats",
        "Observation duration T (days)",
        "Recency (T - t) (days)",
        "Amount of initial purchase ($)"
    ]
)

print("Table 1. Descriptive Statistics for CDNOW dataset")
print(table1_stats.round(2))
display(table1_stats)


# Set the path for the Excel file in the project root's 'excel' folder
excel_path = os.path.join(project_root, "outputs", "excel", "full_summaries.xlsx")
os.makedirs(os.path.dirname(excel_path), exist_ok=True)

# Save the DataFrame to the Excel file
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
    table1_stats.to_excel(writer, sheet_name="Table_1_DescriptStats", index=True)



# %% 4. Compute model fit Bivariate Models
# -- 4. Compute model fit Bivariate Models --

# Function to summarize level 2 draws
def summarize_level2(draws_level2: np.ndarray, param_names: list[str], decimals: int = 2) -> pd.DataFrame:
    quantiles = np.percentile(draws_level2, [2.5, 50, 97.5], axis=0)
    summary = pd.DataFrame(quantiles.T, columns=["2.5%", "50%", "97.5%"], index=param_names)
    return summary.round(decimals)

# Parameter names for Model 1 (M1): no covariates
param_names_bi_m1 = [
    "log_lambda (intercept)",
    "log_mu (intercept)",
    "var_log_lambda",
    "var_log_mu",
    "cov_log_lambda_mu"
]

# Parameter names for Model 2 (M2): with covariate "first.sales"
param_names_bi_m2 = [
    "log_lambda (intercept)",
    "log_lambda (first.sales)",
    "log_lambda (gender)",
    "log_lambda (age)",
    "log_mu (intercept)",
    "log_mu (first.sales)",
    "log_mu (gender)",
    "log_mu (age)",
    "var_log_lambda",
    "var_log_mu",
    "cov_log_lambda_mu"
]

# Compute summaries
summary_bi_m1 = summarize_level2(bi_m1["level_2"][0], param_names=param_names_bi_m1)
summary_bi_m2 = summarize_level2(bi_m2["level_2"][0], param_names=param_names_bi_m2)

# Drop "MAE" row if present
summary_bi_m1 = summary_bi_m1.drop(index="MAE", errors="ignore")
summary_bi_m2 = summary_bi_m2.drop(index="MAE", errors="ignore")

# Rename indices to match Table 3 from the paper
summary_bi_m1.index = [
    "Purchase rate log(λ) - Intercept",
    "Dropout rate log(μ) - Intercept",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma_λ_μ = cov[log λ, log μ]"
] # type: ignore
summary_bi_m2.index = [
    "Purchase rate log(λ) - Intercept",
    "Purchase rate log(λ) - Initial amount ($ 10^-3)",
    "Purchase rate log(λ) - Gender [1 = Male]",
    "Purchase rate log(λ) - Age (scaled)",
    "Dropout rate log(μ) - Intercept",
    "Dropout rate log(μ) - Initial amount ($ 10^-3)",
    "Dropout rate log(μ) - Gender [1 = Male]",
    "Dropout rate log(μ) - Age (scaled)",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma_λ_μ = cov[log λ, log μ]"
] # type: ignore

# ------------------------------------------------------------------

# Compute posterior means of λ and μ
def post_mean_lambdas(draws):
    all_draws = np.concatenate(draws["level_1"], axis=0)
    return all_draws[:, :, 0].mean(axis=0)

def post_mean_mus(draws):
    all_draws = np.concatenate(draws["level_1"], axis=0)
    return all_draws[:, :, 1].mean(axis=0)

# Closed-form expected x_star for validation
t_star = 39.0
mean_lambda_m1 = post_mean_lambdas(bi_m1)
mean_mu_m1     = post_mean_mus(bi_m1)
mean_lambda_m2 = post_mean_lambdas(bi_m2)
mean_mu_m2     = post_mean_mus(bi_m2)

cbs["xstar_m1_pred"] = (mean_lambda_m1/mean_mu_m1) * (1 - np.exp(-mean_mu_m1 * t_star))
cbs["xstar_m2_pred"] = (mean_lambda_m2/mean_mu_m2) * (1 - np.exp(-mean_mu_m2 * t_star))

# Compare MAE
mae_bi_m1 = np.mean(np.abs(cbs["x_star"] - cbs["xstar_m1_pred"]))
mae_bi_m2 = np.mean(np.abs(cbs["x_star"] - cbs["xstar_m2_pred"]))

## The MAE rows are no longer added to the summaries here

# Display both
print("Posterior Summary - Model M1 (no covariates):")
print(summary_bi_m1)

# ------------------------------------------------------------------
print("Posterior Summary - Model M2 (with covariates):")
print(summary_bi_m2)


# %% 5. Construct Table 2: Model Fit Evaluation Bivariate Models
# -- 5. Construct Table 2: Model Fit Evaluation Bivariate Models --
# ------------------------------------------------------------------
# ------ Some prerequisites ------

# Prepare weekly index and counts
first_date = cdnowElog["date"].min()
cdnowElog["week"] = ((cdnowElog["date"] - first_date) // pd.Timedelta("7D")).astype(int) + 1

# Fit classical Pareto/NBD by maximum likelihood
pnbd_mle = ParetoNBDFitter(penalizer_coef=0.0)
pnbd_mle.fit(
    frequency=cbs["x"],
    recency=cbs["t_x"],
    T=cbs["T_cal"]
)
# Classical Pareto/NBD (MLE) expected future repeats for the next 39 weeks
exp_xstar_m1 = pnbd_mle.conditional_expected_number_of_purchases_up_to_time(
    t_star,
    cbs["x"],
    cbs["t_x"],
    cbs["T_cal"]
)

# Set the time range for the analysis
max_week = cdnowElog["week"].max()

times = np.arange(1, max_week + 1)

# Sort the data by customer and week
cdnowElog_sorted = cdnowElog.sort_values(by=["cust","week"])
cdnowElog_sorted["txn_order"] = cdnowElog_sorted.groupby("cust").cumcount()

repeat_txns = cdnowElog_sorted[cdnowElog_sorted["txn_order"] >= 1]

weekly_actual = (
    repeat_txns.groupby("week")["cust"].count()
    .reindex(range(1, max_week+1), fill_value = 0))

cum_pnbd_ml = np.zeros_like(times, dtype=float)

# ------------------------------------------------------------------
# Table 2 – Model‑fit metrics
# ------------------------------------------------------------------
# --- individual‑level correlation & MSE ---------------------------
# Validation period (x_star)

corr_val_pnbd = np.corrcoef(cbs["x_star"], exp_xstar_m1)[0, 1]
mse_val_pnbd  = np.mean((cbs["x_star"] - exp_xstar_m1) ** 2)

corr_val_m1 = np.corrcoef(cbs["x_star"], cbs["xstar_m1_pred"])[0, 1]
mse_val_m1  = np.mean((cbs["x_star"] - cbs["xstar_m1_pred"]) ** 2)

corr_val_m2 = np.corrcoef(cbs["x_star"], cbs["xstar_m2_pred"])[0, 1]
mse_val_m2  = np.mean((cbs["x_star"] - cbs["xstar_m2_pred"]) ** 2)

# Calibration period (x)
# PNB baseline is x itself, so corr=1, mse=0 by definition
corr_calib_pnbd = 1.0
mse_calib_pnbd  = 0.0

calib_pred_m1 = (mean_lambda_m1 / mean_mu_m1) * (1 - np.exp(-mean_mu_m1 * cbs["T_cal"]))
calib_pred_m2 = (mean_lambda_m2 / mean_mu_m2) * (1 - np.exp(-mean_mu_m2 * cbs["T_cal"]))

corr_calib_m1 = np.corrcoef(cbs["x"], calib_pred_m1)[0, 1]
mse_calib_m1  = np.mean((cbs["x"] - calib_pred_m1) ** 2)
corr_calib_m2 = np.corrcoef(cbs["x"], calib_pred_m2)[0, 1]
mse_calib_m2  = np.mean((cbs["x"] - calib_pred_m2) ** 2)

 # Abe (2009): both windows are 39 weeks long – weeks 1‑39 vs. 40‑78
weeks_cal_mask = (times >= 1)  & (times <= 39)
weeks_val_mask = (times >= 40) & (times <= 78)

actual_weekly = weekly_actual.reindex(times, fill_value=0).to_numpy()

def mape_aggregate(actual, pred):
    """
    Abe (2009) time‑series MAPE:
        (1/N) Σ_t |Ĉ(t) − C(t)|  divided by  C(T)   ×100.
    This down‑weights early weeks (matching the paper’s numbers).
    """
    cum_a = np.cumsum(actual)
    cum_p = np.cumsum(pred)
    abs_error = np.abs(cum_p - cum_a)
    return abs_error.mean() / cum_a[-1] * 100

# Weekly PNB (MLE) increments
inc_pnbd_weekly = np.empty_like(times, dtype=float)
inc_pnbd_weekly[0] = cum_pnbd_ml[0]
inc_pnbd_weekly[1:] = np.diff(cum_pnbd_ml)

# def mape_aggregate(actual, pred):
mapecum_val_pnbd = mape_aggregate(actual_weekly[weeks_val_mask], inc_pnbd_weekly[weeks_val_mask])
mapecum_cal_pnbd = mape_aggregate(actual_weekly[weeks_cal_mask], inc_pnbd_weekly[weeks_cal_mask])
mapecum_pool_pnbd = mape_aggregate(actual_weekly, inc_pnbd_weekly)



# Compute hierarchical Bayesian RFM metrics
def compute_metrics(draws: dict, label: str) -> dict[str, float]:
    all_d = np.concatenate(draws["level_1"], axis=0)
    lam = all_d[:, :, 0].mean(axis=0)
    mu  = all_d[:, :, 1].mean(axis=0)
    t_star = 39.0

    # Individual-level repeat forecasts
    xstar_pred = (lam / mu) * (1 - np.exp(-mu * t_star))
    corr_val = np.corrcoef(cbs["x_star"], xstar_pred)[0, 1]
    mse_val  = np.mean((cbs["x_star"] - xstar_pred) ** 2)

    # Calibration forecasts
    calib_pred = (lam / mu) * (1 - np.exp(-mu * cbs["T_cal"]))
    corr_cal = np.corrcoef(cbs["x"], calib_pred)[0, 1]
    mse_cal  = np.mean((cbs["x"] - calib_pred) ** 2)

    # Time-series MAPE via closed-form increments
    lam_over_mu = lam / mu
    cum_expected = np.array([
        np.sum(lam_over_mu * (1 - np.exp(-mu * t)))
        for t in times
    ])
    inc_weekly = np.diff(np.concatenate(([0.0], cum_expected)))
    actual = weekly_actual.to_numpy()
    def _mape(a, p):
        cum_a = np.cumsum(a)
        cum_p = np.cumsum(p)
        return np.abs(cum_p - cum_a).mean() / cum_a[-1] * 100
    mape_val  = _mape(actual[weeks_val_mask], inc_weekly[weeks_val_mask])
    mape_cal  = _mape(actual[weeks_cal_mask], inc_weekly[weeks_cal_mask])
    mape_pool = _mape(actual, inc_weekly)

    return {
        "label":     label,
        "corr_val":  corr_val,
        "corr_cal":  corr_cal,
        "mse_val":   mse_val,
        "mse_cal":   mse_cal,
        "mape_val":  mape_val,
        "mape_cal":  mape_cal,
        "mape_pool": mape_pool
    }

# Gather metrics
stats_pnbd  = {
    "label":     "Pareto/NBD",
    "corr_val":  corr_val_pnbd,
    "corr_cal":  corr_calib_pnbd,
    "mse_val":   mse_val_pnbd,
    "mse_cal":   mse_calib_pnbd,
    "mape_val":  mapecum_val_pnbd,
    "mape_cal":  mapecum_cal_pnbd,
    "mape_pool": mapecum_pool_pnbd
}
stats_bi_m1 = compute_metrics(bi_m1,  "HB M1")
stats_bi_m2 = compute_metrics(bi_m2,  "HB M2")

# Rebuild Table 2 with correct metrics
table2 = pd.DataFrame({
    stats_pnbd["label"]: [
        stats_pnbd[k] for k in [
            "corr_val","corr_cal","mse_val","mse_cal","mape_val","mape_cal","mape_pool"
        ]
    ],
    stats_bi_m1["label"]: [
        stats_bi_m1[k] for k in [
            "corr_val","corr_cal","mse_val","mse_cal","mape_val","mape_cal","mape_pool"
        ]
    ],
    stats_bi_m2["label"]: [
        stats_bi_m2[k] for k in [
            "corr_val","corr_cal","mse_val","mse_cal","mape_val","mape_cal","mape_pool"
        ]
    ]
}, index=[
    "Correlation (Validation)", "Correlation (Calibration)",
    "MSE (Validation)",         "MSE (Calibration)",
    "MAPE (Validation)",        "MAPE (Calibration)", "MAPE (Pooled)"
]).round(2)

# Define row order for display, matching paper layout
metric_order = [
    "Disaggregate measure",
    "Correlation (Validation)", "Correlation (Calibration)", "",
    "MSE (Validation)",         "MSE (Calibration)",         "",
    "Aggregate measure", "Time-series MAPE (%)",
    "MAPE (Validation)",        "MAPE (Calibration)",        "MAPE (Pooled)"
]
# Reapply layout & display
table2 = table2.reindex(metric_order)
table2_formatted = table2.reset_index().rename(columns={"index": ""})
table2_disp = _fmt(table2_formatted, 2)
print("\nTable 2. Model Fit for CDNOW Data")
display(table2_disp)

# ------
# ------ Some prerequisites ------

# Prepare weekly index and counts
first_date = cdnowElog["date"].min()
cdnowElog["week"] = ((cdnowElog["date"] - first_date) // pd.Timedelta("7D")).astype(int) + 1

# Fit classical Pareto/NBD by maximum likelihood
pnbd_mle = ParetoNBDFitter(penalizer_coef=0.0)
pnbd_mle.fit(
    frequency=cbs["x"],
    recency=cbs["t_x"],
    T=cbs["T_cal"]
)
# Classical Pareto/NBD (MLE) expected future repeats for the next 39 weeks
exp_xstar_m1 = pnbd_mle.conditional_expected_number_of_purchases_up_to_time(
    t_star,
    cbs["x"],
    cbs["t_x"],
    cbs["T_cal"]
)

# Set the time range for the analysis
max_week = cdnowElog["week"].max()

times = np.arange(1, max_week + 1)

# Sort the data by customer and week
cdnowElog_sorted = cdnowElog.sort_values(by=["cust","week"])
cdnowElog_sorted["txn_order"] = cdnowElog_sorted.groupby("cust").cumcount()

repeat_txns = cdnowElog_sorted[cdnowElog_sorted["txn_order"] >= 1]

weekly_actual = (
    repeat_txns.groupby("week")["cust"].count()
    .reindex(range(1, max_week+1), fill_value = 0))

cum_pnbd_ml = np.zeros_like(times, dtype=float)

# ------------------------------------------------------------------
# Table 2 – Model‑fit metrics
# ------------------------------------------------------------------
# --- individual‑level correlation & MSE ---------------------------
# Validation period (x_star)

corr_val_pnbd = np.corrcoef(cbs["x_star"], exp_xstar_m1)[0, 1]
mse_val_pnbd  = np.mean((cbs["x_star"] - exp_xstar_m1) ** 2)

corr_val_m1 = np.corrcoef(cbs["x_star"], cbs["xstar_m1_pred"])[0, 1]
mse_val_m1  = np.mean((cbs["x_star"] - cbs["xstar_m1_pred"]) ** 2)

corr_val_m2 = np.corrcoef(cbs["x_star"], cbs["xstar_m2_pred"])[0, 1]
mse_val_m2  = np.mean((cbs["x_star"] - cbs["xstar_m2_pred"]) ** 2)

# Calibration period (x)
# PNB baseline is x itself, so corr=1, mse=0 by definition
corr_calib_pnbd = 1.0
mse_calib_pnbd  = 0.0

calib_pred_m1 = (mean_lambda_m1 / mean_mu_m1) * (1 - np.exp(-mean_mu_m1 * cbs["T_cal"]))
calib_pred_m2 = (mean_lambda_m2 / mean_mu_m2) * (1 - np.exp(-mean_mu_m2 * cbs["T_cal"]))

corr_calib_m1 = np.corrcoef(cbs["x"], calib_pred_m1)[0, 1]
mse_calib_m1  = np.mean((cbs["x"] - calib_pred_m1) ** 2)
corr_calib_m2 = np.corrcoef(cbs["x"], calib_pred_m2)[0, 1]
mse_calib_m2  = np.mean((cbs["x"] - calib_pred_m2) ** 2)

 # Abe (2009): both windows are 39 weeks long – weeks 1‑39 vs. 40‑78
weeks_cal_mask = (times >= 1)  & (times <= 39)
weeks_val_mask = (times >= 40) & (times <= 78)

actual_weekly = weekly_actual.reindex(times, fill_value=0).to_numpy()

def mape_aggregate(actual, pred):
    """
    Abe (2009) time‑series MAPE:
        (1/N) Σ_t |Ĉ(t) − C(t)|  divided by  C(T)   ×100.
    This down‑weights early weeks (matching the paper’s numbers).
    """
    cum_a = np.cumsum(actual)
    cum_p = np.cumsum(pred)
    abs_error = np.abs(cum_p - cum_a)
    return abs_error.mean() / cum_a[-1] * 100

# Weekly PNB (MLE) increments
inc_pnbd_weekly = np.empty_like(times, dtype=float)
inc_pnbd_weekly[0] = cum_pnbd_ml[0]
inc_pnbd_weekly[1:] = np.diff(cum_pnbd_ml)

# def mape_aggregate(actual, pred):
mapecum_val_pnbd = mape_aggregate(actual_weekly[weeks_val_mask], inc_pnbd_weekly[weeks_val_mask])
mapecum_cal_pnbd = mape_aggregate(actual_weekly[weeks_cal_mask], inc_pnbd_weekly[weeks_cal_mask])
mapecum_pool_pnbd = mape_aggregate(actual_weekly, inc_pnbd_weekly)


inc_hb_weekly = np.zeros_like(times, dtype=float)
mapecum_val_m1 = mape_aggregate(actual_weekly[weeks_val_mask], inc_hb_weekly[weeks_val_mask])
mapecum_cal_m1 = mape_aggregate(actual_weekly[weeks_cal_mask], inc_hb_weekly[weeks_cal_mask])
mapecum_pool_m1 = mape_aggregate(actual_weekly, inc_hb_weekly)

# HB M2 uses same weekly draw series
mapecum_val_m2 = mapecum_val_m1
mapecum_cal_m2 = mapecum_cal_m1
mapecum_pool_m2 = mapecum_pool_m1

# ------------------------------------------------------------------
# --- assemble DataFrame ------------------------------------------
# ------------------------------------------------------------------

table2 = pd.DataFrame({
    "Pareto/NBD": [corr_val_pnbd, corr_calib_pnbd,
                   mse_val_pnbd,  mse_calib_pnbd,
                   mapecum_val_pnbd, mapecum_cal_pnbd, mapecum_pool_pnbd],
    "HB M1":      [corr_val_m1,   corr_calib_m1,
                   mse_val_m1,    mse_calib_m1,
                   mapecum_val_m1,   mapecum_cal_m1,  mapecum_pool_m1],
    "HB M2":      [corr_val_m2,   corr_calib_m2,
                   mse_val_m2,    mse_calib_m2,
                   mapecum_val_m2,   mapecum_cal_m2,  mapecum_pool_m2],
}, index=[
    "Correlation (Validation)", "Correlation (Calibration)",
    "MSE (Validation)",         "MSE (Calibration)",
    "MAPE (Validation)",        "MAPE (Calibration)", "MAPE (Pooled)"
]).round(2)

# ---- re‑format Table 2 rows to match paper layout --------------------------
metric_order = [
    "Disaggregate measure",
    "Correlation (Validation)", "Correlation (Calibration)", "",
    "MSE (Validation)",         "MSE (Calibration)",         "",
    "Aggregate measure", "Time-series MAPE (%)",
    "MAPE (Validation)",        "MAPE (Calibration)",        "MAPE (Pooled)"
]
table2 = table2.reindex(metric_order)

# Convert index into a column so the left‑most column shows the metric label
table2_formatted = table2.reset_index().rename(columns={"index": ""})

# ---- non‑coloured Table 2 display and save -------------------------------
table2_disp = _fmt(table2_formatted, 2)
print("\nTable 2. Model Fit for CDNOW Data")
display(table2_disp)

# Save to Excel
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    table2_formatted.to_excel(writer, sheet_name="Table 2", index=False, float_format="%.2f")
# %% 6. Construct Table 2: Model Fit Evaluation Trivariate Models
# -- 6. Construct Table 2: Model Fit Evaluation Trivariate Models --
# ----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Build helper arrays (weeks, masks, birth_week) – run once ----------
# -----------------------------------------------------------------------

cbs_df = cbs


first_date = cdnowElog["date"].min()
cdnowElog["week"] = ((cdnowElog["date"] - first_date)
                     // pd.Timedelta("7D")).astype(int) + 1
max_week = cdnowElog["week"].max()

elog_sorted = cdnowElog.sort_values(["cust", "week"])
elog_sorted["txn_order"] = elog_sorted.groupby("cust").cumcount()
repeat_txns = elog_sorted[elog_sorted["txn_order"] >= 1]

weekly_actual = (
    repeat_txns.groupby("week")["cust"].count()
               .reindex(range(1, max_week + 1), fill_value=0)
)

times = np.arange(1, max_week + 1)
birth_week = (
    cdnowElog.groupby("cust")["week"].min()
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
        tau_d = draws["level_1"][ch][idx, :, 2]

        rng = np.random.default_rng(d)
        for t_idx, t in enumerate(times):
            active = (t > birth_week) & (t <= (birth_week + tau_d))
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
stats_0 = compute_metrics(tri_m1,  "HB RFM (no cov)")
stats_1 = compute_metrics(tri_m2, "HB RFM (+ gender & age)")

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
print("\nModel fit for Trivariate models")
display(table2_disp)

# %% 7. Figure 2: Weekly-Series Tracking
# -- 7. Figure 2: Weekly-Series Tracking --
# -------------------------------------------------------------------
# Figure 2: Weekly cumulative repeat transactions for all models
# ------------------------------------------------------------------
# Cumulative actual transactions
cum_actual = weekly_actual.cumsum()

# Pareto/NBD (MLE) baseline
cum_pnbd_ml = np.zeros_like(times, dtype=float)
for t_idx, t in enumerate(times):
    rel_t = np.clip(t - birth_week, 0, None)
    exp_per_cust = pnbd_mle.expected_number_of_purchases_up_to_time(rel_t)
    cum_pnbd_ml[t_idx] = exp_per_cust.sum()

# Bivariate HB M1 and M2 closed-form cumulative forecasts
lam_bi1 = mean_lambda_m1
mu_bi1  = mean_mu_m1
lam_bi2 = mean_lambda_m2
mu_bi2  = mean_mu_m2

cum_bi1 = np.array([
    np.sum((lam_bi1/mu_bi1)*(1 - np.exp(-mu_bi1 * t)))
    for t in times
])
cum_bi2 = np.array([
    np.sum((lam_bi2/mu_bi2)*(1 - np.exp(-mu_bi2 * t)))
    for t in times
])

# Trivariate HB (intercept-only and + gender & age)
def simulate_hb_cumulative(draws):
    all_chains = draws["level_1"]
    total_draws = sum(chain.shape[0] for chain in all_chains)
    inc = np.zeros_like(times, dtype=float)
    for chain_idx, chain in enumerate(all_chains):
        for draw_idx in range(chain.shape[0]):
            lam_d = chain[draw_idx, :, 0]
            mu_d  = chain[draw_idx, :, 1]
            tau_d = chain[draw_idx, :, 2]
            rng = np.random.default_rng(chain_idx * chain.shape[0] + draw_idx)
            for t_idx, t in enumerate(times):
                active = (t > birth_week) & (t <= birth_week + tau_d)
                inc[t_idx] += rng.poisson(lam=lam_d * active).sum()
    inc /= total_draws
    return np.cumsum(inc)

cum_tri1 = simulate_hb_cumulative(tri_m1)
cum_tri2 = simulate_hb_cumulative(tri_m2)

# Plot all curves
plt.figure(figsize=(8,5))
plt.plot(times, cum_actual, '-', label="Actual")
plt.plot(times, cum_pnbd_ml, '--', label="Pareto/NBD (MLE)")
plt.plot(times, cum_bi1, ':', label="HB Bivariate M1")
plt.plot(times, cum_bi2, '-.', label="HB Bivariate M2")
plt.plot(times, cum_tri1, '-', label="HB Trivariate M1")
plt.plot(times, cum_tri2, '--', label="HB Trivariate M2")
plt.axvline(x=t_star, color='k', linestyle='--')
plt.xlabel("Week")
plt.ylabel("Cumulative repeat transactions")
plt.title("Figure 2: Weekly Time-Series Tracking for CDNOW Data")
plt.legend()
plt.savefig(os.path.join(save_figures_path, "Figure2_weekly_tracking.png"), dpi=300, bbox_inches='tight')
plt.show()


plt.savefig(os.path.join(project_root, "outputs", "figures","Figure3_conditional_expectation.png"), dpi=300, bbox_inches='tight')

# %% 8. Figure 3: Conditional expectation of future transactions for all models
# -------------------------------------------------------------------
# Compute per‐customer expected future transactions at t_star

# 1) Pareto/NBD (MLE)
exp_pnbd = exp_xstar_m1

# 2) HB Bivariate M1
exp_bi1 = (mean_lambda_m1 / mean_mu_m1) * (1 - np.exp(-mean_mu_m1 * t_star))

# 3) HB Bivariate M2
exp_bi2 = (mean_lambda_m2 / mean_mu_m2) * (1 - np.exp(-mean_mu_m2 * t_star))

# 4) HB Trivariate M1
all_tri1 = np.concatenate(tri_m1["level_1"], axis=0)
lam_tri1 = all_tri1[:, :, 0].mean(axis=0)
mu_tri1  = all_tri1[:, :, 1].mean(axis=0)
z_tri1   = all_tri1[:, :, 3].mean(axis=0)
exp_tri1 = z_tri1 * (lam_tri1 / mu_tri1) * (1 - np.exp(-mu_tri1 * t_star))

# 5) HB Trivariate M2
all_tri2 = np.concatenate(tri_m2["level_1"], axis=0)
lam_tri2 = all_tri2[:, :, 0].mean(axis=0)
mu_tri2  = all_tri2[:, :, 1].mean(axis=0)
z_tri2   = all_tri2[:, :, 3].mean(axis=0)
exp_tri2 = z_tri2 * (lam_tri2 / mu_tri2) * (1 - np.exp(-mu_tri2 * t_star))

# Assemble into DataFrame
df_cond = pd.DataFrame({
    "x":            cbs["x"],
    "Actual":       cbs["x_star"],
    "Pareto/NBD":   exp_pnbd,
    "HB Bi M1":     exp_bi1,
    "HB Bi M2":     exp_bi2,
    "HB Tri M1":    exp_tri1,
    "HB Tri M2":    exp_tri2
})

# Group by calibration count (0–7+)
groups = []
for k in range(7):
    grp = df_cond[df_cond["x"] == k]
    groups.append((
        str(k),
        grp["Actual"].mean(),
        grp["Pareto/NBD"].mean(),
        grp["HB Bi M1"].mean(),
        grp["HB Bi M2"].mean(),
        grp["HB Tri M1"].mean(),
        grp["HB Tri M2"].mean()
    ))
grp7 = df_cond[df_cond["x"] >= 7]
groups.append((
    "7+",
    grp7["Actual"].mean(),
    grp7["Pareto/NBD"].mean(),
    grp7["HB Bi M1"].mean(),
    grp7["HB Bi M2"].mean(),
    grp7["HB Tri M1"].mean(),
    grp7["HB Tri M2"].mean()
))
cond_df = pd.DataFrame(groups, columns=[
    "x", "Actual", "Pareto/NBD", "HB Bi M1", "HB Bi M2", "HB Tri M1", "HB Tri M2"
]).set_index("x")

# Plot conditional expectations
plt.figure(figsize=(8,5))
plt.plot(cond_df.index, cond_df["Actual"],       '-',  label="Actual")
plt.plot(cond_df.index, cond_df["Pareto/NBD"],   '--', label="Pareto/NBD")
plt.plot(cond_df.index, cond_df["HB Bi M1"],     ':',  label="HB Bivariate M1")
plt.plot(cond_df.index, cond_df["HB Bi M2"],     '-.', label="HB Bivariate M2")
plt.plot(cond_df.index, cond_df["HB Tri M1"],    '-',  label="HB Trivariate M1")
plt.plot(cond_df.index, cond_df["HB Tri M2"],    '--', label="HB Trivariate M2")
plt.xlabel("Transactions in calibration (weeks 1–39)")
plt.ylabel("Average transactions in weeks 40–78")
plt.title("Figure 3: Conditional Expectation of Future Transactions")
plt.legend()
plt.savefig(
    os.path.join(save_figures_path, "Figure3_conditional_expectation_all_models.png"),
    dpi=300, bbox_inches='tight'
)
plt.show()

# %% 9. MCMC Diagnostics
# -- 9. MCMC Diagnostics --

# -------------------------------------------------------------------
# Plot 6. Create scatterplots to visualize the predictions of both models
# -------------------------------------------------------------------
import seaborn as sns
sns.set(style="whitegrid")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Scatterplot for Model M1
axes[0].scatter(cbs["x_star"], cbs["xstar_m1_pred"], alpha=0.4, color="tab:blue")
axes[0].plot([0, cbs["x_star"].max()], [0, cbs["x_star"].max()], 'r--')
axes[0].set_title("M1: Without Covariates")
axes[0].set_xlabel("Actual x_star")
axes[0].set_ylabel("Predicted x_star")

# Scatterplot for Model M2
axes[1].scatter(cbs["x_star"], cbs["xstar_m2_pred"], alpha=0.4, color="tab:green")
axes[1].plot([0, cbs["x_star"].max()], [0, cbs["x_star"].max()], 'r--')
axes[1].set_title("M2: With first.sales")
axes[1].set_xlabel("Actual x_star")

# Remove grid from both subplots
for ax in axes:
    ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(project_root, "outputs", "figures","Scatter_M1_M2.png"), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# 7. Visualize the predicted alive vs. churned customers
# -------------------------------------------------------------------
# Add a new column for predicted alive status based on xstar_m2_pred
cbs["is_alive_pred"] = np.where(cbs["xstar_m2_pred"] >= 1, 1, 0)

# Prepare data
counts = cbs["is_alive_pred"].value_counts().sort_index()
labels = ["Churned (z = 0)", "Alive (z = 1)"]
colors = ["#d3d3d3", "#4a90e2"]  # Light grey and business blue

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 4))

# Plot bar chart
bars = ax.bar(labels, counts, color=colors, width=0.5)

# Set axis labels and title
ax.set_ylabel("Number of customers", fontsize=11)
ax.set_title("Predicted Alive vs. Churned\n(Last Draw of MCMC Chain)", fontsize=13)


# Annotate each bar with its value
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        f"{int(height)}",
        ha='center',
        va='bottom',
        fontsize=10
    )

# Disable grid lines so they don’t appear behind annotations
ax.grid(False)

# Clean up the axis appearance
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Hide the vertical left spine so it doesn’t bisect the first bar
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#999999')
ax.tick_params(axis='y', colors='#444444')
ax.tick_params(axis='x', colors='#444444')

# Final layout adjustment
plt.tight_layout()
plt.savefig(os.path.join(project_root, "outputs", "figures","Alive_vs_Churned.png"), dpi=300, bbox_inches='tight')
plt.show()
# -------------------------------------------------------------------
# 9. MCMC Diagnostics for all four models
# -------------------------------------------------------------------
import arviz as az

# Convert Bivariate M1 to InferenceData
idata_bi_m1 = az.from_dict(
    posterior={"level_2": np.array(bi_m1["level_2"])},
    coords={"param": param_names_bi_m1},
    dims={"level_2": ["param"]}
)

# Convert Bivariate M2 to InferenceData
idata_bi_m2 = az.from_dict(
    posterior={"level_2": np.array(bi_m2["level_2"])},
    coords={"param": param_names_bi_m2},
    dims={"level_2": ["param"]}
)

# Convert Trivariate M1 to InferenceData
idata_tri_m1 = az.from_dict(
    posterior={"level_2": np.array(tri_m1["level_2"])},
    coords={"param": param_names_tri_m1},
    dims={"level_2": ["param"]}
)

# Convert Trivariate M2 to InferenceData
idata_tri_m2 = az.from_dict(
    posterior={"level_2": np.array(tri_m2["level_2"])},
    coords={"param": param_names_tri_m2},
    dims={"level_2": ["param"]}
)

# Traceplots
for idata, label in [
    (idata_bi_m1, "HB Bivariate M1"),
    (idata_bi_m2, "HB Bivariate M2"),
    (idata_tri_m1, "HB Trivariate M1"),
    (idata_tri_m2, "HB Trivariate M2"),
]:
    az.plot_trace(idata, var_names=["level_2"], figsize=(12, max(4, len(idata.posterior["level_2"].coords["param"]) * 1.5)))
    plt.suptitle(f"Traceplot - {label}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# Autocorrelation plots
for idata, label in [
    (idata_bi_m1, "HB Bivariate M1"),
    (idata_bi_m2, "HB Bivariate M2"),
    (idata_tri_m1, "HB Trivariate M1"),
    (idata_tri_m2, "HB Trivariate M2"),
]:
    az.plot_autocorr(idata, var_names=["level_2"], figsize=(12, max(4, len(idata.posterior["level_2"].coords["param"]) * 1.5)))
    plt.suptitle(f"Autocorrelation - {label}", fontsize=14)
    plt.tight_layout()
    plt.show()

# Posterior distributions
for idata, label in [
    (idata_bi_m1, "HB Bivariate M1"),
    (idata_bi_m2, "HB Bivariate M2"),
    (idata_tri_m1, "HB Trivariate M1"),
    (idata_tri_m2, "HB Trivariate M2"),
]:
    n_params = len(idata.posterior["level_2"].coords["param"])
    az.plot_posterior(
        idata,
        var_names=["level_2"],
        figsize=(8, n_params * 2),
        hdi_prob=0.95,
        kind='kde',
        grid=(n_params, 1)
    )
    plt.suptitle(f"Posterior Distributions - {label}", fontsize=16, y=1.02)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

# %%
