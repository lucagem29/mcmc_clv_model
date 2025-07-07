# ------------------------------------------------------------------
# this script reproduces the analysis from Abe (2009)
# ------------------------------------------------------------------
# %% 1. Import necessary libraries & set project root & custom modules & helper function
# -- 1. Import necessary libraries & set project root & custom modules & helper function --
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
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pandas as pd
import pickle

# Add lifetimes ParetoNBDFitter for MLE baseline
from lifetimes import ParetoNBDFitter  
from IPython.display import display

# Import custom functions 
from src.models.bivariate.mcmc import draw_future_transactions
from src.models.utils.analysis_display_helper import _fmt
from src.models.utils.analysis_bi_helpers import (summarize_level2, 
                                                  post_mean_lambdas, 
                                                  post_mean_mus, 
                                                  mape_aggregate, 
                                                  extract_correlation, 
                                                  chain_total_loglik, 
                                                  compute_table4)
from src.models.utils.analysis_bi_dynamic import build_bivariate_param_names_and_labels

# %% 2. Load estimates and data 
# ------------------------------------------------------------------
# --- Load Pre-computed Results ---
pickles_dir = os.path.join(project_root, "outputs", "pickles")

# Set Excel output path
excel_path = os.path.join(project_root, "outputs", "excel", "abe_replication.xlsx")
os.makedirs(os.path.dirname(excel_path), exist_ok=True)

# Load MCMC draws
with open(os.path.join(pickles_dir, "full_bi_m1.pkl"), "rb") as f:
    draws_m1 = pickle.load(f)
with open(os.path.join(pickles_dir, "full_bi_m2.pkl"), "rb") as f:
    draws_m2 = pickle.load(f)

# Load CBS data ---> CHNAGE TO FULL ONCE ALL WORKS
cbs_path = os.path.join(project_root, "data", "processed", "cdnow_abeCBS.csv")
print(f"Loading CBS data from: {cbs_path}")
cbs = pd.read_csv(cbs_path, dtype={"cust": str}, parse_dates=["first"])

# Load Elog data ---> CHNAGE TO FULL ONCE ALL WORKS
data_path = os.path.join(project_root, "data", "raw", "cdnow_abeElog.csv")
print(f"Loading Elog data from: {data_path}")
cdnowElog = pd.read_csv(data_path)
# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])
# ensure the same key type
cdnowElog["cust"] = cdnowElog["cust"].astype(str)

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
            cdnowElog.groupby("cust")["sales"].first().mean(),
            cbs["age"].mean(),
            cbs["gender_binary"].mean(),
        ],
        "Std. dev.": [
            cbs["x"].std(),
            cbs["T_cal"].std() * 7,
            (cbs["T_cal"] - cbs["t_x"]).std() * 7,
            cdnowElog.groupby("cust")["sales"].first().std(),
            cbs["age"].std(),
            cbs["gender_binary"].std(),
        ],
        "Min": [
            cbs["x"].min(),
            cbs["T_cal"].min() * 7,
            (cbs["T_cal"] - cbs["t_x"]).min() * 7,
            cdnowElog.groupby("cust")["sales"].first().min(),
            cbs["age"].min(),
            cbs["gender_binary"].min(),
        ],
        "Max": [
            cbs["x"].max(),
            cbs["T_cal"].max() * 7,
            (cbs["T_cal"] - cbs["t_x"]).max() * 7,
            cdnowElog.groupby("cust")["sales"].first().max(),
            cbs["age"].max(),
            cbs["gender_binary"].max(),
        ],
    },
    index=[
        "Number of repeats",
        "Observation duration T (days)",
        "Recency (T - t) (days)",
        "Amount of initial purchase ($)",
        "Age",
        "Gender (0: F | 1: M)"
    ]
)

print("Table 1. Descriptive Statistics for CDNOW dataset")
print(table1_stats.round(2))
display(table1_stats)

# Save the DataFrame to the Excel file
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
    table1_stats.to_excel(writer, sheet_name="Table 1")
# ------------------------------------------------------------------
# %%

# Parameter names and labels for Model 1 (M1): no covariates
param_names_m1, labels_m1 = build_bivariate_param_names_and_labels([])

# Parameter names and labels for Model 2 (M2): with covariates
covariate_cols_m2 = ["first.sales", "age_scaled", "gender_binary"]
param_names_m2, labels_m2 = build_bivariate_param_names_and_labels(covariate_cols_m2)

# Compute summaries
summary_m1 = summarize_level2(draws_m1["level_2"][0], param_names=param_names_m1)
summary_m2 = summarize_level2(draws_m2["level_2"][0], param_names=param_names_m2)

# Drop "MAE" row if present
summary_m1 = summary_m1.drop(index="MAE", errors="ignore")
summary_m2 = summary_m2.drop(index="MAE", errors="ignore")

# Rename indices to match Table 3 from the paper (now dynamic)
summary_m1.index = labels_m1  # type: ignore
summary_m2.index = labels_m2  # type: ignore

# ------------------------------------------------------------------
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
    "Aggregate measure",        "Time-series MAPE (%)",
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

corr_m1 = extract_correlation(np.array(draws_m1["level_2"][0]))
corr_m2 = extract_correlation(np.array(draws_m2["level_2"][0]))

# Create correlation DataFrame
correlation_row = pd.DataFrame({
    ("HB M1 (no covariates)", "2.5%"): [corr_m1[0]],
    ("HB M1 (no covariates)", "50%"): [corr_m1[1]],
    ("HB M1 (no covariates)", "97.5%"): [corr_m1[2]],
    ("HB M2 (with 3 covariates)", "2.5%"): [corr_m2[0]],
    ("HB M2 (with 3 covariates)", "50%"): [corr_m2[1]],
    ("HB M2 (with 3 covariates)", "97.5%"): [corr_m2[2]],
}, index=["Correlation computed from Γ₀"])

ll_m1 = chain_total_loglik(draws_m1["level_1"], cbs).round(0)
ll_m2 = chain_total_loglik(draws_m2["level_1"], cbs).round(0)

loglik_row = pd.DataFrame({
    ("HB M1 (no covariates)", "2.5%"): [""],
    ("HB M1 (no covariates)", "50%"):  [round(ll_m1, 0)],
    ("HB M1 (no covariates)", "97.5%"): [""],
    ("HB M2 (with 3 covariates)", "2.5%"): [""],
    ("HB M2 (with 3 covariates)", "50%"):  [round(ll_m2, 0)],
    ("HB M2 (with 3 covariates)", "97.5%"): [""],
}, index=["Marginal log-likelihood"])

# Format summary into 2D (col=quantiles) with aligned indices
summary_m1_cleaned = summary_m1.copy()
summary_m2_cleaned = summary_m2.copy()

# Align the summaries vertically using dynamic labels from the helper
row_labels = labels_m2

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
    [["HB M1 (no covariates)", "HB M2 (with 3 covariates)"], ["2.5%", "50%", "97.5%"]]
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
# Reset inc_hb_weekly to zero for Figure 2 calculation
inc_hb_weekly = np.zeros_like(times, dtype=float)

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

# --- DEBUG PRINTS FOR FIGURE 2 ---
print("\n[DEBUG] Figure 2 Data Preview:")
print("cum_actual:", cum_actual[:10])
print("cum_pnbd_ml:", cum_pnbd_ml[:10])
print("cum_hb:", cum_hb[:10])
print("times:", times[:10])
print("Lengths:")
print("  cum_actual:", len(cum_actual))
print("  cum_pnbd_ml:", len(cum_pnbd_ml))
print("  cum_hb:", len(cum_hb))
print("  times:", len(times))

# --- END DEBUG PRINTS ---

plt.figure(figsize=(8,5))
plt.plot(times, cum_actual, '-', color='tab:blue', linewidth=2, label="Actual")
plt.plot(times, cum_pnbd_ml, '--', color='tab:orange', linewidth=2, label="Pareto/NBD (MLE)")
plt.plot(times, cum_hb, ':', color='tab:green', linewidth=2, label="HB")
plt.axvline(x=int(t_star), color='k', linestyle='--')
plt.xlabel("Week")
plt.ylabel("Cumulative repeat transactions")
plt.title("Figure 2: Weekly Time-Series Tracking for CDNOW Data")
plt.legend()
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_extention", "Figure2_weekly_tracking.png"), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_extention", "Figure3_conditional_expectation.png"), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_extention", "Figure4_scatter_lambda_mu.png"),
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
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_extention", "Figure5_corr_histogram.png"),
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
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_extention", "Scatter_M1_M2.png"), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(project_root, "outputs", "figures", "full_extention", "Alive_vs_Churned.png"), dpi=300, bbox_inches='tight')
plt.show()
# -------------------------------------------------------------------
# 8. Visualize the posterior distributions and traceplots for both models
# -------------------------------------------------------------------
# Convert M1 to InferenceData
idata_m1 = az.from_dict(
    posterior={"level_2": np.array(draws_m1["level_2"])},  # shape: (chains, draws, dims)
    coords={"param": param_names_m1},
    dims={"level_2": ["param"]}
)
# Convert M2 to InferenceData
idata_m2 = az.from_dict(
    posterior={"level_2": np.array(draws_m2["level_2"])},
    coords={"param": param_names_m2},
    dims={"level_2": ["param"]}
)

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