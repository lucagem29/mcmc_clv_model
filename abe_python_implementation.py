# %% 1. Import necessary libraries
# ------ 1. Import necessary libraries ------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import os
from openpyxl import load_workbook
# Add lifetimes ParetoNBDFitter for MLE baseline
from lifetimes import ParetoNBDFitter

# Import functions 
# Import custom modules for data processing
from Models.elog2cbs import elog2cbs

# Import custom modules for model estimation and prediction
from Models.pareto_abe_manual import (
    mcmc_draw_parameters,
    draw_future_transactions
)

# Ensure Estimation directory exists
os.makedirs("Estimation", exist_ok=True)
excel_path = "Estimation/estimation_summaries.xlsx"

# %% 2. Load dataset and convert to CBS format
# ------ 2. Load dataset and convert to CBS format ------
# We use dataset available in the BTYD package in R
data_path = os.path.join("Data", "cdnowElog.csv")
cdnowElog = pd.read_csv(data_path)

# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])

# Convert event log to customer-by-sufficient-statistic (CBS) format
cbs = elog2cbs(cdnowElog, units="W", T_cal="1997-09-30", T_tot="1998-06-30")
#cbs = create_customer_summary(cdnowElog, T_cal="1997-09-30", T_tot="1998-06-30")
cbs = cbs.rename(columns={"t.x": "t_x", "T.cal": "T_cal", "x.star": "x_star"})
cbs.head()

# %% 3. Construct Table 1: Descriptive Statistics
# ------ Construct Table 1 from Abe 2009 (Descriptive Statistics) ---
# Compute statistics: mean, std, min, max for each of the four fields
table1_stats = pd.DataFrame({
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
    ]
}, index=[
    "Number of repeats",
    "Observation duration T (days)",
    "Recency (T - t) (days)",
    "Amount of initial purchase ($)"
])

print("Table 1. Descriptive Statistics for CDNOW dataset")
print(table1_stats.round(2))

# Save both summaries to a single Excel file with two sheets
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
    table1_stats.to_excel(writer, sheet_name="Table 1")

# %% 4. Estimate Model M1 (no covariates)
# ------ 3. Start with model estimation M1 using no covariates ------

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

# ------ 4. Estimate Model M2 (with covariates) ------

# Append dollar amount of first purchase to use as covariate (like in R)
first = cdnowElog.groupby("cust")["sales"].first().reset_index()
first["first.sales"] = first["sales"] * 1e-3
cbs = pd.merge(cbs, first[["cust", "first.sales"]], on="cust", how="left")

# Normalize first.sales
mean_val = cbs["first.sales"].mean()
std_val = cbs["first.sales"].std()
cbs["first.sales_scaled"] = (cbs["first.sales"] - mean_val) / std_val

# Estimate Model M2 (with first.sales)
draws_m2 = mcmc_draw_parameters(
    cal_cbs=cbs,
    covariates=["first.sales_scaled"],
    mcmc=4000,
    burnin=10000,
    thin=50,
    chains=2,
    seed=42,
    trace=500
)

# %% 5. Compute metrics and predictions
# ------ 5. Computing the metrics for comparison ------

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
]
summary_m2.index = [
    "Purchase rate log(λ) - Intercept",
    "Purchase rate log(λ) - Initial amount ($ 10^-3)",
    "Dropout rate log(μ) - Intercept",
    "Dropout rate log(μ) - Initial amount ($ 10^-3)",
    "sigma^2_λ = var[log λ]",
    "sigma^2_μ = var[log μ]",
    "sigma_λ_μ = cov[log λ, log μ]"
]

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

# %% 6. Construct Table 2: Model Fit Evaluation
# ------ Construct Table 2: Model Fit Evaluation ------

# Validation correlation (individual-level)
corr_val_m1 = np.corrcoef(cbs["x_star"], cbs["xstar_m1_pred"])[0, 1]
corr_val_m2 = np.corrcoef(cbs["x_star"], cbs["xstar_m2_pred"])[0, 1]

# Calibration correlation (actual = x, predicted = model expectation using posterior mean of λ and μ)

calib_pred_m1 = (mean_lambda_m1 / mean_mu_m1) * (1 - np.exp(-mean_mu_m1 * cbs["T_cal"]))
calib_pred_m2 = (mean_lambda_m2 / mean_mu_m2) * (1 - np.exp(-mean_mu_m2 * cbs["T_cal"]))
corr_calib_m1 = np.corrcoef(cbs["x"], calib_pred_m1)[0, 1]
corr_calib_m2 = np.corrcoef(cbs["x"], calib_pred_m2)[0, 1]

# Compute MSE
mse_val_m1 = np.mean((cbs["x_star"] - cbs["xstar_m1_pred"])**2)
mse_val_m2 = np.mean((cbs["x_star"] - cbs["xstar_m2_pred"])**2)
mse_calib_m1 = np.mean((cbs["x"] - calib_pred_m1)**2)
mse_calib_m2 = np.mean((cbs["x"] - calib_pred_m2)**2)


# ------ Weekly-aggregated MAPE as in Abe (2009) ------
def weekly_mape(actual, pred):
    mask = actual > 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

# Anchor weeks at the start of the dataset
cal_start_date = cdnowElog["date"].min()
cdnowElog["week_idx"] = ((cdnowElog["date"] - cal_start_date) // pd.Timedelta("7D")).astype(int) + 1

cal_weeks = int(t_star)
val_weeks = int(t_star)

# Actual weekly counts for calibration
actual_cal = (
    cdnowElog[cdnowElog["week_idx"] <= cal_weeks]
    .groupby("week_idx")["cust"]
    .count()
    .reindex(range(1, cal_weeks+1), fill_value=0)
    .to_numpy()
)
# Predicted weekly counts for calibration
weeks_cal = np.arange(1, cal_weeks+1)
inc_cal_m1 = (mean_lambda_m1[:, None] / mean_mu_m1[:, None]) * (
    np.exp(-mean_mu_m1[:, None] * (weeks_cal - 1))
    - np.exp(-mean_mu_m1[:, None] * weeks_cal)
)
pred_cal_m1 = inc_cal_m1.sum(axis=0)
inc_cal_m2 = (mean_lambda_m2[:, None] / mean_mu_m2[:, None]) * (
    np.exp(-mean_mu_m2[:, None] * (weeks_cal - 1))
    - np.exp(-mean_mu_m2[:, None] * weeks_cal)
)
pred_cal_m2 = inc_cal_m2.sum(axis=0)

# Actual weekly counts for validation
actual_val = (
    cdnowElog[(cdnowElog["week_idx"] > cal_weeks) & (cdnowElog["week_idx"] <= cal_weeks+val_weeks)]
    .groupby("week_idx")["cust"]
    .count()
    .reindex(range(cal_weeks+1, cal_weeks+val_weeks+1), fill_value=0)
    .to_numpy()
)
# Predicted weekly counts for validation
weeks_val = np.arange(cal_weeks+1, cal_weeks+val_weeks+1)
inc_val_m1 = (mean_lambda_m1[:, None] / mean_mu_m1[:, None]) * (
    np.exp(-mean_mu_m1[:, None] * (weeks_val-1))
    - np.exp(-mean_mu_m1[:, None] * weeks_val)
)
pred_val_m1 = inc_val_m1.sum(axis=0)
inc_val_m2 = (mean_lambda_m2[:, None] / mean_mu_m2[:, None]) * (
    np.exp(-mean_mu_m2[:, None] * (weeks_val-1))
    - np.exp(-mean_mu_m2[:, None] * weeks_val)
)
pred_val_m2 = inc_val_m2.sum(axis=0)

# Compute weekly MAPE
mape_cal_m1 = weekly_mape(actual_cal, pred_cal_m1)
mape_cal_m2 = weekly_mape(actual_cal, pred_cal_m2)
mape_val_m1 = weekly_mape(actual_val, pred_val_m1)
mape_val_m2 = weekly_mape(actual_val, pred_val_m2)
# Pooled
actual_pooled = np.concatenate([actual_cal, actual_val])
pred_pooled_m1 = np.concatenate([pred_cal_m1, pred_val_m1])
pred_pooled_m2 = np.concatenate([pred_cal_m2, pred_val_m2])
mape_pooled_m1 = weekly_mape(actual_pooled, pred_pooled_m1)
mape_pooled_m2 = weekly_mape(actual_pooled, pred_pooled_m2)

# Construct Table 2 with weekly-aggregated MAPE
table2 = pd.DataFrame({
    "HB M1": [corr_val_m1, corr_calib_m1, mse_val_m1, mse_calib_m1, mape_val_m1, mape_cal_m1, mape_pooled_m1],
    "HB M2": [corr_val_m2, corr_calib_m2, mse_val_m2, mse_calib_m2, mape_val_m2, mape_cal_m2, mape_pooled_m2]
}, index=[
    "Correlation (Validation)", "Correlation (Calibration)",
    "MSE (Validation)", "MSE (Calibration)",
    "MAPE (Validation)", "MAPE (Calibration)", "MAPE (Pooled)"
])

with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    table2.to_excel(writer, sheet_name="Table 2")

# %% 7. Construct Table 3: Estimation Results
# ------ Construct Table 3 from Abe 2009 (Estimation Results) ------
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

# Create Marginal Log-Likelihood row
loglik_row = pd.DataFrame({
    ("HB M1 (no covariates)", "2.5%"): [""],
    ("HB M1 (no covariates)", "50%"): [-abs(round(draws_m1["log_likelihood"], 3))],
    ("HB M1 (no covariates)", "97.5%"): [""],
    ("HB M2 (with a covariate)", "2.5%"): [""],
    ("HB M2 (with a covariate)", "50%"): [-abs(round(draws_m2["log_likelihood"], 3))],
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

# Save the table
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    table3_combined.to_excel(writer, sheet_name="Table 3")

# %% 8. Construct Table 4: Customer-Level Statistics
# ------ Construct Table 4: Customer-Level Statistics ------
# Generate posterior predictive draws for validation period
xstar_m1_draws = draw_future_transactions(cbs, draws_m1, T_star=t_star, seed=42)
xstar_m2_draws = draw_future_transactions(cbs, draws_m2, T_star=t_star, seed=42)

def compute_table4(draws, xstar_draws):
    # Average over all level_1 draws from all chains
    all_draws = np.concatenate(draws["level_1"], axis=0)  # shape: (n_draws, n_customers, 4)
    
    # Compute posterior means across all draws for each parameter
    mean_lambda = all_draws[:, :, 0].mean(axis=0)
    mean_mu = all_draws[:, :, 1].mean(axis=0)
    mean_z = all_draws[:, :, 3].mean(axis=0)
    # Expected x_star based on Equation (8) from Abe (2009), with t = 39 weeks
    t_star = 39
    mean_xstar = mean_lambda / mean_mu * (1 - np.exp(-mean_mu * t_star))

    # Formula (9): Expected lifetime = 1 / μ
    mean_lifetime = np.where(mean_mu > 0, 1.0 / mean_mu, np.inf)

    # Formula (10): 1-year survival rate = exp(-52 * μ), where 52 weeks = 1 year
    surv_1yr = np.exp(-mean_mu * 52)

    # Compute posterior percentiles for each parameter
    lambda_draws = all_draws[:, :, 0]
    mu_draws = all_draws[:, :, 1]

    lambda_2_5 = np.percentile(lambda_draws, 2.5, axis=0)
    lambda_97_5 = np.percentile(lambda_draws, 97.5, axis=0)
    mu_2_5 = np.percentile(mu_draws, 2.5, axis=0)
    mu_97_5 = np.percentile(mu_draws, 97.5, axis=0)

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
    return df.round(3)

table4 = compute_table4(draws_m2, xstar_m2_draws)

# Save both new tables
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    table4.to_excel(writer, sheet_name="Table 4")


# %% Figures 2–5: Reproduce Abe (2009) plots
# Prepare weekly index and counts
first_date = cdnowElog["date"].min()
cdnowElog["week"] = ((cdnowElog["date"] - first_date) // pd.Timedelta("7D")).astype(int) + 1
max_week = cdnowElog["week"].max()

# Fit classical Pareto/NBD by maximum likelihood
pnbd_mle = ParetoNBDFitter(penalizer_coef=0.0)
pnbd_mle.fit(
    frequency=cbs["x"],
    recency=cbs["t_x"],
    T=cbs["T_cal"]
)

# Figure 2: Weekly cumulative repeat transactions
weekly_actual = cdnowElog.groupby("week")["cust"].count().reindex(range(1, max_week+1), fill_value=0)
cum_actual = weekly_actual.cumsum()

# True Pareto/NBD (MLE) expected cumulative transactions (aggregated)
times = np.arange(1, max_week+1)
# expected cumulative purchases per customer at each t
per_cust_pnbd = np.array([
    pnbd_mle.expected_number_of_purchases_up_to_time(t) for t in times
])
# multiply by number of customers to get group-level cumulative
cum_pnbd_ml = per_cust_pnbd * len(cbs)

# HB model cumulative
inc_hb = (mean_lambda_m2[:, None]/mean_mu_m2[:, None]) * (
    np.exp(-mean_mu_m2[:, None]*(times-1)) - np.exp(-mean_mu_m2[:, None]*times)
)
cum_hb = inc_hb.sum(axis=0).cumsum()

plt.figure(figsize=(8,5))
plt.plot(times, cum_actual, '-', color='tab:blue', linewidth=2, label="Actual")
plt.plot(times, cum_pnbd_ml, '--', color='tab:orange', linewidth=2, label="Pareto/NBD (MLE)")
plt.plot(times, cum_hb, ':', color='tab:green', linewidth=2, label="HB")
plt.axvline(x=int(t_star), color='k', linestyle='--')
plt.xlabel("Week")
plt.ylabel("Cumulative repeat transactions")
plt.title("Figure 2: Weekly Time-Series Tracking for CDNOW Data")
plt.legend()
plt.savefig(os.path.join("Estimation","Figure2_weekly_tracking.png"), dpi=300, bbox_inches='tight')
plt.show()

#
# Figure 3: Conditional expectation of future transactions
# Group by number of calibration transactions (0–7+)
# Updated to use Pareto/NBD baseline with Model M1 posterior means
df = pd.DataFrame({
    "x": cbs["x"],
    "actual": cbs["x_star"],
    "pnbd": (mean_lambda_m1/mean_mu_m1)*(1-np.exp(-mean_mu_m1*cbs["T_cal"])),
    "hb": cbs["xstar_m2_pred"]
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
plt.savefig(os.path.join("Estimation","Figure3_conditional_expectation.png"), dpi=300, bbox_inches='tight')
plt.show()

# Figure 4: Scatter plot of posterior means of λ and μ
plt.figure(figsize=(6,6))
plt.scatter(mean_lambda_m2, mean_mu_m2, alpha=0.3)
plt.xlabel("λ")
plt.ylabel("μ")
plt.title("Figure 4: Scatter Plot of Posterior Means of λ and μ for CDNOW Data")
plt.savefig(os.path.join("Estimation","Figure4_scatter_lambda_mu.png"), dpi=300, bbox_inches='tight')
plt.show()

# Figure 5: Histogram of correlation between log(λ) and log(μ)
# Compute correlation draws from level_2
corr_draws = []
for chain in draws_m2["level_2"]:
    for draw in chain:
        cov = draw[-2]; var_l = draw[-3]; var_m = draw[-1]
        corr_draws.append(cov/np.sqrt(var_l*var_m))
plt.figure(figsize=(8,4))
plt.hist(corr_draws, bins=20, edgecolor='k')
plt.xlabel("Correlation")
plt.ylabel("Frequency")
plt.title("Figure 5: Distribution of Correlation Between log(λ) and log(μ) for CDNOW Data")
plt.savefig(os.path.join("Estimation","Figure5_corr_histogram.png"), dpi=300, bbox_inches='tight')
plt.show()










# %% ------------ Additional visualizations and diagnostics ------------

# 6. Create scatterplots to visualize the predictions of both models
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
plt.savefig(os.path.join("Estimation","Scatter_M1_M2.png"), dpi=300, bbox_inches='tight')
plt.show()

# 7. Visualize the predicted alive vs. churned customers
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
plt.savefig(os.path.join("Estimation","Alive_vs_Churned.png"), dpi=300, bbox_inches='tight')
plt.show()

# 8. Visualize the posterior distributions and traceplots for both models
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

# 10. Autocorrelation – M1 vs. M2
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
# %%
