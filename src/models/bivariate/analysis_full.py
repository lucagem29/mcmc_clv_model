# ------------------------------------------------------------------
# this script extends the analysis from Abe (2009) with the whole CDNOW dataset
# ------------------------------------------------------------------

# %% 1. Import necessary libraries & set project root & custom modules & helper function
# -- 1. Import necessary libraries & set project root & custom modules & helper function --
# ------------------------------------------------------------------
import sys
import os
from src.utils.project_root import add_project_root_to_sys_path
project_root = add_project_root_to_sys_path()
# ------------------------------------------------------------------
# ------------------------------------------------------------------

# Import rest of libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pandas as pd
import pickle

# Add lifetimes ParetoNBDFitter for MLE baseline
from lifetimes import ParetoNBDFitter

from scipy.special import gammaln  # for log‑factorial constant
from IPython.display import display

# Import custom module 
from src.models.bivariate.mcmc import draw_future_transactions

# ---------------------------------------------------------------------
# Helper: enforce uniform decimal display (e.g. 0.63, 2.57, …)
# ---------------------------------------------------------------------
def _fmt(df: pd.DataFrame, dec: int) -> pd.DataFrame:
    """Return a copy of *df* with all float cells formatted to *dec* decimals."""
    fmt = f"{{:.{dec}f}}".format
    return df.applymap(lambda v: fmt(v) if isinstance(v, (float, np.floating)) else v)
# ------------------------------------------------------------------
# %% 2. Load estimates and data
# -- 2. Load estimates and data --
# ------------------------------------------------------------------
# --- Load Pre-computed Results ---
pickles_dir = os.path.join(project_root, "outputs", "pickles")

# Load MCMC draws
with open(os.path.join(pickles_dir, "full_bivariate_M1.pkl"), "rb") as f:
    draws_m1 = pickle.load(f)
with open(os.path.join(pickles_dir, "full_bivariate_M2.pkl"), "rb") as f:
    draws_m2 = pickle.load(f)

# Load the CBS DataFrame
with open(os.path.join(pickles_dir, "cbs_full_bivariate_data.pkl"), "rb") as f:
    cbs = pickle.load(f)

data_path = os.path.join(project_root, "data", "raw", "cdnow_purchases.csv")

cdnowElog = pd.read_csv(data_path)
# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])
# ------------------------------------------------------------------
# %% 3. Descriptive Statistics
# -- 3. Descriptive Statistics --
# ------------------------------------------------------------------
# ------ Construct Table 1 from Abe 2009  ------
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

print("Table 1. Descriptive Statistics for full CDNOW dataset")
print(table1_stats.round(2))
display(table1_stats)


# Set the path for the Excel file in the project root's 'excel' folder
excel_path = os.path.join(project_root, "outputs", "excel", "full_bivariate_estimation_summaries.xlsx")
os.makedirs(os.path.dirname(excel_path), exist_ok=True)

# Save the DataFrame to the Excel file
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
    table1_stats.to_excel(writer, sheet_name="Table 1")
# ------------------------------------------------------------------
# %% 4. Compute metrics and predictions
# -- 4. Compute metrics and predictions --
# ------------------------------------------------------------------

# Function to summarize level 2 draws
def summarize_level2(draws_level2: np.ndarray, param_names: list[str], decimals: int = 2) -> pd.DataFrame:
    quantiles = np.percentile(draws_level2, [2.5, 50, 97.5], axis=0)
    summary = pd.DataFrame(quantiles.T, columns=["2.5%", "50%", "97.5%"], index=param_names)
    return summary.round(decimals)

# Parameter names for Model 1 (M1): no covariates
param_names_m1 = [
    "log_lambda (intercept)",
    "log_mu (intercept)",
    "var_log_lambda",
    "var_log_mu",
    "cov_log_lambda_mu"
]

# Parameter names for Model 2 (M2): with covariate "first.sales"
param_names_m2 = [
    "log_lambda (intercept)",
    "log_lambda (first.sales)",
    "log_mu (intercept)",
    "log_mu (first.sales)",
    "var_log_lambda",
    "var_log_mu",
    "cov_log_lambda_mu"
]

# Compute summaries
summary_m1 = summarize_level2(draws_m1["level_2"][0], param_names=param_names_m1)
summary_m2 = summarize_level2(draws_m2["level_2"][0], param_names=param_names_m2)

# Drop "MAE" row if present
summary_m1 = summary_m1.drop(index="MAE", errors="ignore")
summary_m2 = summary_m2.drop(index="MAE", errors="ignore")

# Rename indices to match Table 3 from the paper
summary_m1.index = [
    "Purchase rate log(λ) - Intercept",
    "Dropout rate log(μ) - Intercept",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma_λ_μ = cov[log λ, log μ]"
] # type: ignore
summary_m2.index = [
    "Purchase rate log(λ) - Intercept",
    "Purchase rate log(λ) - Initial amount ($ 10^-3)",
    "Dropout rate log(μ) - Intercept",
    "Dropout rate log(μ) - Initial amount ($ 10^-3)",
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
mean_lambda_m1 = post_mean_lambdas(draws_m1)
mean_mu_m1     = post_mean_mus(draws_m1)
mean_lambda_m2 = post_mean_lambdas(draws_m2)
mean_mu_m2     = post_mean_mus(draws_m2)

cbs["xstar_m1_pred"] = (mean_lambda_m1/mean_mu_m1) * (1 - np.exp(-mean_mu_m1 * t_star))
cbs["xstar_m2_pred"] = (mean_lambda_m2/mean_mu_m2) * (1 - np.exp(-mean_mu_m2 * t_star))

# Compare MAE
mae_m1 = np.mean(np.abs(cbs["x_star"] - cbs["xstar_m1_pred"]))
mae_m2 = np.mean(np.abs(cbs["x_star"] - cbs["xstar_m2_pred"]))

## The MAE rows are no longer added to the summaries here

# Display both
print("Posterior Summary - Model M1 (no covariates):")
print(summary_m1)

print("Posterior Summary - Model M2 (with covariates):")
print(summary_m2)
# ------------------------------------------------------------------
# %% 5. Construct Table 2: Model Fit Evaluation
# -- 5. Construct Table 2: Model Fit Evaluation --
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
# ------------------------------------------------------------------
# %% 6. Construct Table 3: Estimation Results
# -- 6. Construct Table 3: Estimation Results --
# ------------------------------------------------------------------
# --- Compute correlation between log_lambda and log_mu from posterior (for both models) ---
def extract_correlation(draws_level2):
    cov = draws_level2[:, -2]  # cov_log_lambda_mu
    var_lambda = draws_level2[:, -3]
    var_mu = draws_level2[:, -1]
    corr = cov / np.sqrt(var_lambda * var_mu)
    return np.percentile(corr, [2.5, 50, 97.5]).round(2)

corr_m1 = extract_correlation(np.array(draws_m1["level_2"][0]))
corr_m2 = extract_correlation(np.array(draws_m2["level_2"][0]))

# Create correlation DataFrame
correlation_row = pd.DataFrame({
    ("HB M1 (no covariates)", "2.5%"): [corr_m1[0]],
    ("HB M1 (no covariates)", "50%"): [corr_m1[1]],
    ("HB M1 (no covariates)", "97.5%"): [corr_m1[2]],
    ("HB M2 (with a covariate)", "2.5%"): [corr_m2[0]],
    ("HB M2 (with a covariate)", "50%"): [corr_m2[1]],
    ("HB M2 (with a covariate)", "97.5%"): [corr_m2[2]],
}, index=["Correlation computed from Γ₀"])

# --- compute chain‑averaged log‑likelihood ---------------------------------
def chain_total_loglik(level1_chains, cbs):
    """Return average over draws of Σ_i log L_i."""
    x = cbs["x"].to_numpy()
    T_cal = cbs["T_cal"].to_numpy()
    totals = []
    for chain in level1_chains:
        for draw in chain:            # draw shape (N, 4)
            lam = draw[:, 0]
            mu  = draw[:, 1]
            tau = draw[:, 2]
            z   = draw[:, 3] > 0.5
            ll_vec = (
                x * np.log(lam)
                + (1 - z) * np.log(mu)
                - (lam + mu) * (z * T_cal + (1 - z) * tau)
                - gammaln(x + 1)            # remove constant term for comparability
            )
            totals.append(ll_vec.sum())
    return np.mean(totals)

ll_m1 = chain_total_loglik(draws_m1["level_1"], cbs).round(0)
ll_m2 = chain_total_loglik(draws_m2["level_1"], cbs).round(0)

loglik_row = pd.DataFrame({
    ("HB M1 (no covariates)", "2.5%"): [""],
    ("HB M1 (no covariates)", "50%"):  [round(ll_m1, 0)],
    ("HB M1 (no covariates)", "97.5%"): [""],
    ("HB M2 (with a covariate)", "2.5%"): [""],
    ("HB M2 (with a covariate)", "50%"):  [round(ll_m2, 0)],
    ("HB M2 (with a covariate)", "97.5%"): [""],
}, index=["Marginal log-likelihood"])

# Format summary into 2D (col=quantiles) with aligned indices
summary_m1_cleaned = summary_m1.copy()
summary_m2_cleaned = summary_m2.copy()

# Align the summaries vertically with correct row structure
row_labels = [
    "Purchase rate log(λ) - Intercept",
    "Purchase rate log(λ) - Initial amount ($ 10^-3)",
    "Dropout rate log(μ) - Intercept",
    "Dropout rate log(μ) - Initial amount ($ 10^-3)",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma_λ_μ = cov[log λ, log μ]"
]

# Create placeholder rows for missing M1 covariates
m1_fill = pd.DataFrame(index=row_labels, columns=["2.5%", "50%", "97.5%"])
for idx in summary_m1_cleaned.index:
    m1_fill.loc[idx] = summary_m1_cleaned.loc[idx]

m2_fill = pd.DataFrame(index=row_labels, columns=["2.5%", "50%", "97.5%"])
for idx in summary_m2_cleaned.index:
    m2_fill.loc[idx] = summary_m2_cleaned.loc[idx]

# Concatenate horizontally for final Table 3 view
table3_combined = pd.concat([m1_fill, m2_fill], axis=1)
table3_combined.columns = pd.MultiIndex.from_product(
    [["HB M1 (no covariates)", "HB M2 (with a covariate)"], ["2.5%", "50%", "97.5%"]]
)

# Append the correlation row and loglik row to Table 3
table3_combined = pd.concat([table3_combined, correlation_row, loglik_row])

# Display Table 3
display(_fmt(table3_combined, 2))

# Save the table
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    table3_combined.to_excel(writer, sheet_name="Table 3")
# ------------------------------------------------------------------
# %% 7. Construct Table 4: Customer-Level Statistics
# -- 7. Construct Table 4: Customer-Level Statistics --
# -------------------------------------------------------------------
# Generate posterior predictive draws for validation period
xstar_m1_draws = draw_future_transactions(cbs, draws_m1, T_star=t_star, seed=42)
xstar_m2_draws = draw_future_transactions(cbs, draws_m2, T_star=t_star, seed=42)

def compute_table4(draws, xstar_draws):
    # Average over all level_1 draws from all chains
    all_draws = np.concatenate(draws["level_1"], axis=0)  # shape: (n_draws, n_customers, 4)
    
    # Compute posterior means across all draws for each parameter
    mean_lambda = all_draws[:, :, 0].mean(axis=0)

    # --- posterior μ --------------------------------------------------------
    # Use a light cap (0.05) when averaging to prevent a few extreme draws
    mu_draws_raw = all_draws[:, :, 1]
    mu_draws_cap = np.clip(mu_draws_raw, None, 0.05)   # gentle cap for means
    # Posterior **mean** of μ uses the capped draws to dampen a few extremes,
    # but the uncertainty intervals (2.5 %, 97.5 %) must be computed on the
    # *raw* draws – otherwise the upper tail is artificially flattened.
    mean_mu = mu_draws_cap.mean(axis=0)

    mu_2_5  = np.percentile(mu_draws_raw,  2.5, axis=0)
    mu_97_5 = np.percentile(mu_draws_raw, 97.5, axis=0)

    mean_z = all_draws[:, :, 3].mean(axis=0)
    t_star = 39
    # Expected repeats in validation window *unconditional* on survival:
    #   E[X*] = P(alive at T_cal) · λ/μ · (1 - e^{-μ·t})
    mean_xstar = (
        mean_z
        * (mean_lambda / mean_mu)
        * (1.0 - np.exp(-mean_mu * t_star))
    )

    # Formula (9): Expected lifetime = 1 / μ, convert weeks to years (divide by 52)
    mean_lifetime = np.where(mean_mu > 0, (1.0 / mean_mu) / 52.0, np.inf)

    # Formula (10): 1-year survival rate = exp(-52 * μ), where 52 weeks = 1 year
    surv_1yr = np.exp(-mean_mu * 52)

    # Posterior percentiles for λ (unchanged); μ percentiles now from uncapped draws
    lambda_draws = all_draws[:, :, 0]
    lambda_2_5 = np.percentile(lambda_draws, 2.5, axis=0)
    lambda_97_5 = np.percentile(lambda_draws, 97.5, axis=0)

    df = pd.DataFrame({
        "Mean(λ)": mean_lambda,
        "2.5% tile λ": lambda_2_5,
        "97.5% tile λ": lambda_97_5,
        "Mean(μ)": mean_mu,
        "2.5% tile μ": mu_2_5,
        "97.5% tile μ": mu_97_5,
        "Mean exp lifetime (yrs)": mean_lifetime,
        "Survival rate (1yr)": surv_1yr,
        "P(alive at T_cal)": mean_z,
        "Exp # of trans in val period": mean_xstar
    })
    df.index.name = "Customer ID"

    # Rank customers by expected transactions (high → low) and
    # assign sequential IDs 1…N exactly like Abe (2009)
    df_sorted = (
        df.sort_values("Exp # of trans in val period",
                       ascending=False)          # high → low, paper order
          .reset_index(drop=True)                # throw away original cust nums
    )
    df_sorted.insert(0, "ID", df_sorted.index + 1)   # 1‑based rank ID
    top10    = df_sorted.iloc[:10]
    bottom10 = df_sorted.iloc[-10:]
    ave_row  = df.mean().to_frame().T.assign(ID="Ave")
    min_row  = df.min().to_frame().T.assign(ID="Min")
    max_row  = df.max().to_frame().T.assign(ID="Max")

    # Concatenate in paper order
    df_paper = (
        pd.concat([top10, pd.DataFrame({"ID":["…"]}), bottom10,
                   ave_row, min_row, max_row], ignore_index=True)
          .set_index("ID")
    )

    # --- column‑specific rounding to match Abe (2009) print layout ----------
    lambda_cols = ["Mean(λ)", "2.5% tile λ", "97.5% tile λ"]
    mu_cols     = ["Mean(μ)", "2.5% tile μ", "97.5% tile μ"]

    df_paper[lambda_cols] = df_paper[lambda_cols].round(3)   # e.g. 0.778
    df_paper[mu_cols]     = df_paper[mu_cols].round(4)       # e.g. 0.0187

    df_paper["Mean exp lifetime (yrs)"]     = df_paper["Mean exp lifetime (yrs)"].round(2)
    df_paper["Survival rate (1yr)"]         = df_paper["Survival rate (1yr)"].round(3)
    df_paper["P(alive at T_cal)"]           = df_paper["P(alive at T_cal)"].round(3)
    df_paper["Exp # of trans in val period"] = df_paper["Exp # of trans in val period"].round(2)

    return df_paper


table4 = compute_table4(draws_m2, xstar_m2_draws)

# Show Table 4 exactly as rounded inside `compute_table4`
display(table4)
# Save both new tables
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    table4.to_excel(writer, sheet_name="Table 4")
# -------------------------------------------------------------------
# %% 8. Figures 2–5: Reproduce Abe (2009) plots
# -- 8. Figures 2–5: Reproduce Abe (2009) plots --
# -------------------------------------------------------------------
# Figure 2: Weekly cumulative repeat transactions
# -------------------------------------------------------------------
# Cumulative actual transactions
cum_actual = weekly_actual.cumsum()

 # --- Birth‑aligned Pareto/NBD baseline (MLE) -----------------
# first purchase week for each customer
birth_week = (
    cdnowElog.groupby("cust")["week"].min()
    .reindex(cbs["cust"])
    .to_numpy()
)

for t_idx, t in enumerate(times):
    # time since first purchase (≥0) for each customer
    rel_t = np.clip(t - birth_week, 0, None)
    exp_per_cust = pnbd_mle.expected_number_of_purchases_up_to_time(rel_t)
    cum_pnbd_ml[t_idx] = exp_per_cust.sum()

# --- Posterior‑predictive HB curve -----------------------------------------
n_draws = len(xstar_m2_draws)

for d in range(n_draws):
    # map flat draw index `d` to (chain, draw) indices
    draws_per_chain = len(draws_m2["level_1"][0])
    chain = d // draws_per_chain
    idx   = d % draws_per_chain
    lam_d = draws_m2["level_1"][chain][idx, :, 0]
    mu_d  = draws_m2["level_1"][chain][idx, :, 1]
    tau_d = draws_m2["level_1"][chain][idx, :, 2]

    rng_d = np.random.default_rng(d)  # reproducible per draw
    for t_idx, t in enumerate(times):
        dt = 1.0
        active = (t > birth_week) & (t <= (birth_week + tau_d))   # after first purchase, before churn
        inc = rng_d.poisson(lam=lam_d * dt * active)
        inc_hb_weekly[t_idx] += inc.sum()

# average across draws and take cumulative
inc_hb_weekly /= n_draws
cum_hb = np.cumsum(inc_hb_weekly)

plt.figure(figsize=(8,5))
plt.plot(times, cum_actual, '-', color='tab:blue', linewidth=2, label="Actual")
plt.plot(times, cum_pnbd_ml, '--', color='tab:orange', linewidth=2, label="Pareto/NBD (MLE)")
plt.plot(times, cum_hb, ':', color='tab:green', linewidth=2, label="HB")
plt.axvline(x=int(t_star), color='k', linestyle='--')
plt.xlabel("Week")
plt.ylabel("Cumulative repeat transactions")
plt.title("Figure 2: Weekly Time-Series Tracking for CDNOW Data")
plt.legend()
plt.savefig(os.path.join(project_root, "outputs", "figures","full_cdnow", "Figure2_weekly_tracking.png"), dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# Figure 3: Conditional expectation of future transactions
# -------------------------------------------------------------------
# Group by number of calibration transactions (0–7+)
# Use analytical expectations, with different formulas for Pareto/NBD (M1) and HB (M2)

# Expected future repeats for Figure 3:
all_draws_m1 = np.concatenate(draws_m1["level_1"], axis=0)
all_draws_m2 = np.concatenate(draws_m2["level_1"], axis=0)

mean_lambda_m1_cust = all_draws_m1[:, :, 0].mean(axis=0)
mean_mu_m1_cust     = all_draws_m1[:, :, 1].mean(axis=0)
mean_z_m1_cust      = all_draws_m1[:, :, 3].mean(axis=0)

mean_lambda_m2_cust = all_draws_m2[:, :, 0].mean(axis=0)
mean_mu_m2_cust     = all_draws_m2[:, :, 1].mean(axis=0)
mean_z_m2_cust      = all_draws_m2[:, :, 3].mean(axis=0)

# HB expectation (Model M2) – include posterior P(alive)
exp_xstar_m2 = mean_z_m2_cust * (mean_lambda_m2_cust / mean_mu_m2_cust) * (1 - np.exp(-mean_mu_m2_cust * t_star))

df = pd.DataFrame({
    "x":      cbs["x"],
    "actual": cbs["x_star"],
    "pnbd":   exp_xstar_m1,   # Pareto/NBD expectation (no P(alive))
    "hb":     exp_xstar_m2    # HB expectation (with P(alive))
})
groups = []
for k in range(7):
    grp = df[df["x"]==k]
    groups.append((str(k), grp["actual"].mean(), grp["pnbd"].mean(), grp["hb"].mean()))
grp7 = df[df["x"]>=7]
groups.append(("7+", grp7["actual"].mean(), grp7["pnbd"].mean(), grp7["hb"].mean()))
cond_df = pd.DataFrame(groups, columns=["x","Actual","Pareto/NBD","HB"]).set_index("x")

plt.figure(figsize=(8,5))
plt.plot(cond_df.index, cond_df["Actual"], '-', color='tab:blue', linewidth=2, label="Actual")
plt.plot(cond_df.index, cond_df["Pareto/NBD"], marker='*', linestyle='--', color='tab:orange', linewidth=2, label="Pareto/NBD")
plt.plot(cond_df.index, cond_df["HB"], marker='x', linestyle=':', color='tab:green', linewidth=2, label="HB")
plt.xlabel("Number of transactions in weeks 1–39")
plt.ylabel("Average transactions in weeks 40–78")
plt.title("Figure 3: Conditional Expectation of Future Transactions for CDNOW Data")
plt.legend()
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_cdnow", "Figure3_conditional_expectation.png"), dpi=300, bbox_inches='tight')
plt.show()
# -------------------------------------------------------------------
# Figure 4: Scatter plot of posterior means of λ and μ  (HB‑M1, paper style)
# -------------------------------------------------------------------
mean_lambda_m1 = post_mean_lambdas(draws_m1)
mean_mu_m1     = post_mean_mus(draws_m1)

plt.figure(figsize=(6, 6))
plt.scatter(mean_lambda_m1, mean_mu_m1, s=8, alpha=0.25, color="tab:blue")
plt.xlim(0, 4)
plt.ylim(0, 0.14)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\mu$")
plt.title("Figure 4: Scatter Plot of Posterior Means of λ and μ for CDNOW Data")
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_cdnow", "Figure4_scatter_lambda_mu.png"),
            dpi=300, bbox_inches="tight")
plt.show()
# -------------------------------------------------------------------
# Figure 5: Histogram of correlation between log(λ) and log(μ)
# -------------------------------------------------------------------

# Flatten level‑2 draws across chains
level2_all = np.vstack(draws_m2["level_2"])   # shape (total_draws, n_params)

# Column order in draws: [..., var_log_lambda, var_log_mu, cov_log_lambda_mu]
var_l = level2_all[:, -3]   # sigma^2_lambda
cov   = level2_all[:, -2]   # cov_log_lambda_mu
var_m = level2_all[:, -1]   # sigma^2_mu

# Keep draws with strictly positive variances
mask = (var_l > 0) & (var_m > 0)
corr_draws = cov[mask] / np.sqrt(var_l[mask] * var_m[mask])

plt.figure(figsize=(8, 4))
plt.hist(corr_draws, bins=30, edgecolor="k")
plt.xlim(-0.3, 0.4)
plt.xlabel("Correlation")
plt.ylabel("Frequency")
plt.title("Figure 5: Distribution of Correlation Between log(λ) and log(μ) for CDNOW Data")
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_cdnow", "Figure5_corr_histogram.png"),
            dpi=300, bbox_inches="tight")
plt.show()
# -------------------------------------------------------------------
# %% 9. Additional visualizations and diagnostics 
# -- 9. Additional visualizations and diagnostics --
# -------------------------------------------------------------------
# Plot 6. Create scatterplots to visualize the predictions of both models
# -------------------------------------------------------------------
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
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_cdnow", "Scatter_M1_M2.png"), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_cdnow", "Alive_vs_Churned.png"), dpi=300, bbox_inches='tight')
plt.show()
# -------------------------------------------------------------------
# 8. Visualize the posterior distributions and traceplots for both models
# -------------------------------------------------------------------
# Convert M1 to InferenceData
idata_m1 = az.from_dict(
    posterior={"level_2": np.array(draws_m1["level_2"])},  # shape: (chains, draws, dims)
    coords={"param": [  # labels for better plots
        "log_lambda (intercept)", 
        "log_mu (intercept)", 
        "var_log_lambda", 
        "cov_log_lambda_mu", 
        "var_log_mu"
    ]},
    dims={"level_2": ["param"]}
)
# Convert M2 to InferenceData
idata_m2 = az.from_dict(
    posterior={"level_2": np.array(draws_m2["level_2"])},
    coords={"param": [
        "log_lambda (intercept)",
        "log_lambda (first.sales)",
        "log_mu (intercept)",
        "log_mu (first.sales)",
        "var_log_lambda",
        "cov_log_lambda_mu",
        "var_log_mu"
    ]},
    dims={"level_2": ["param"]}
)

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Plot traceplots for both models
az.plot_trace(idata_m1, var_names=["level_2"], figsize=(12, 6))
plt.suptitle("Traceplot - M1", fontsize=14)
plt.tight_layout()
plt.show()

az.plot_trace(idata_m2, var_names=["level_2"], figsize=(12, 10))
plt.suptitle("Traceplot - M2", fontsize=14)
plt.tight_layout()
plt.show()

# 9. Summary and convergence diagnostics
# Traceplot – M1 and M2
az.summary(idata_m1, var_names=["level_2"], round_to=4)

# Convergence Summary – M1
az.summary(idata_m2, var_names=["level_2"], round_to=4)

# Convergence Summary – M2
# For M1
az.plot_autocorr(idata_m1, var_names=["level_2"], figsize=(12, 6))
plt.suptitle("Autocorrelation - M1", fontsize=14)
plt.tight_layout()
plt.show()

# For M2
az.plot_autocorr(idata_m2, var_names=["level_2"], figsize=(12, 10))
plt.suptitle("Autocorrelation - M2", fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 10. Autocorrelation – M1 vs. M2
# -------------------------------------------------------------------
# Get number of parameters (last dimension)
n_params_m1 = idata_m1.posterior["level_2"].shape[-1]
n_params_m2 = idata_m2.posterior["level_2"].shape[-1]

# M1
fig = az.plot_posterior(
    idata_m1,
    var_names=["level_2"],
    figsize=(8, n_params_m1 * 2),  # auto-height
    hdi_prob=0.95,
    kind='kde',
    grid=(n_params_m1, 1)  # 1 column, n rows
)
plt.suptitle("Posterior Distributions - M1", fontsize=16, y=1.02)
plt.subplots_adjust(hspace=0.5)
plt.show()

# M2
fig = az.plot_posterior(
    idata_m2,
    var_names=["level_2"],
    figsize=(8, n_params_m2 * 2),  # auto-height
    hdi_prob=0.95,
    kind='kde',
    grid=(n_params_m2, 1)
)
plt.suptitle("Posterior Distributions - M2", fontsize=16, y=1.02)
plt.subplots_adjust(hspace=0.5)
plt.show()
# -------------------------------------------------------------------