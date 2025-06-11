# %% [Imports]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

import arviz as az
from lifetimes import ParetoNBDFitter

from Models.abe_hb_pareto_nbd import AbeHBParetoNBD, load_cdnow, CustomerRFT

sns.set_theme(style="whitegrid")

# %% ------------------------- [Load CDNOW]
customers, _ = load_cdnow()                      # downloads once → ~/.cache
txn = pd.read_csv(
    "~/.cache/cdnow_master.txt",
    sep=r"\s+",
    header=0,
    names=["id", "date", "qty", "spend"],
    parse_dates=["date"],
)
txn["t_week"] = (
    txn["date"] - txn.groupby("id")["date"].transform("min")
).dt.days // 7

print(f"{len(customers):,} customers loaded.")

# %% ------------------------- [Prepare RFM summary for lifetimes]
calib_weeks = 39
summary = (
    txn[txn["t_week"] <= calib_weeks]
    .groupby("id")
    .agg(
        frequency=("id", lambda x: len(x) - 1),
        recency=("t_week", "max"),
        T=("t_week", lambda x: calib_weeks),
    )
)
freq, reci, Tarr = summary["frequency"].values, summary["recency"].values, summary["T"].values

# %% ------------------------- [Fit models]
# Pareto/NBD (population)
pnbd = ParetoNBDFitter()
pnbd.fit(freq, reci * 7, Tarr * 7)               # lifetimes wants *days*

# HB M1 (no covariate)
hb_m1 = AbeHBParetoNBD(customers, seed=42)
hb_m1.run_mcmc(n_iter=14_000, burn_in=10_000, thin=1, proposal_scale=0.25)

# HB M2 (covariate = initial spend scaled $1k)
init_spend = (
    txn.sort_values("date").groupby("id")["spend"].first().reindex(summary.index).values / 1000
)
from abe_hb_pareto_nbd import CustomerRFT  # ensures symbol exists even if imports cell not run
cust_cov = [
    CustomerRFT(x=f, t_x=r, T=t, d=np.array([1.0, s]))
    for f, r, t, s in zip(freq, reci, Tarr, init_spend)
]
hb_m2 = AbeHBParetoNBD(cust_cov, seed=123)
hb_m2.run_mcmc(n_iter=14_000, burn_in=10_000, thin=1, proposal_scale=0.25)

# Helper for HB expected repeats
def hb_expected(hb, weeks):
    lam, mu = hb.posterior_means()
    w = np.asarray(weeks, dtype=float)
    return (lam[:, None] / mu[:, None]) * (1 - np.exp(-mu[:, None] * w))

# %% ------------------------- [Figure 2]
def figure2_tracking(model_hb, pnbd, txn, calib_weeks=39, horizon_weeks=80):
    weeks = np.arange(0, horizon_weeks + 1)

    # Actual repeats (exclude first purchase)
    txn_c = txn.copy()
    txn_c["rank"] = txn_c.sort_values("date").groupby("id").cumcount()
    repeats = txn_c[txn_c["rank"] > 0]
    actual = (
        repeats[repeats["t_week"] <= horizon_weeks]
        .groupby("t_week").size()
        .reindex(weeks, fill_value=0)
        .cumsum()
    )

    # Pareto/NBD curve
    cum_pnbd = np.array([
        pnbd.conditional_expected_number_of_purchases_up_to_time(w * 7, freq, reci * 7, Tarr * 7).sum()
        for w in weeks
    ])

    # HB curve
    cum_hb = hb_expected(model_hb, weeks).sum(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(weeks, actual,  label="Actual",      lw=2, color="k")
    ax.plot(weeks, cum_pnbd, label="Pareto/NBD", ls="--", lw=2, color="tab:blue")
    ax.plot(weeks, cum_hb,   label="HB",         ls=":",  lw=2, color="tab:orange")
    ax.axvline(calib_weeks, ls="--", color="grey")
    ax.set(xlim=(0, horizon_weeks), ylim=(0, None),
           xlabel="Week", ylabel="Cumulative repeat transactions",
           title="Figure 2 — Weekly Time‑Series Tracking Plot for CDNOW Data")
    ax.legend(frameon=False)
    sns.despine(fig)
    return fig

figure2_tracking(hb_m1, pnbd, txn)

# %% ------------------------- [Figure 3 — Conditional expectation plot]
def figure3_conditional(model_hb, pnbd, txn, calib_weeks=39):
    # Count calibration‐window repeat purchases per customer
    x = (
        txn[(txn["t_week"] <= calib_weeks)]
        .groupby("id").size().sub(1).clip(lower=0)   # repeat count
        .reindex(summary.index, fill_value=0)
    )
    bins = np.append(np.arange(0, 7), [99])          # 0‑6, 7+
    groups = np.digitize(x, bins) - 1

    # actual future repeats (weeks 40‑78)
    future = (
        txn[(txn["t_week"] > calib_weeks) & (txn["t_week"] <= 78)]
        .groupby("id").size()
        .reindex(summary.index, fill_value=0)
    )

    def avg_future(model):
        exp_39_78 = hb_expected(model, 78)[:, -1] - hb_expected(model, 39)[:, 0]
        return pd.Series(exp_39_78).groupby(groups).mean()

    act = future.groupby(groups).mean()
    pn  = pd.Series(
        pnbd.conditional_expected_number_of_purchases_up_to_time(78*7, freq, reci*7, Tarr*7)
        - pnbd.conditional_expected_number_of_purchases_up_to_time(39*7, freq, reci*7, Tarr*7)
    ).groupby(groups).mean()
    hb  = avg_future(model_hb)

    idx = ["0", "1", "2", "3", "4", "5", "6", "7+"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(idx, act, "-o", label="Actual")
    ax.plot(idx, pn,  "-o", label="Pareto/NBD")
    ax.plot(idx, hb,  "-o", label="HB")
    ax.set(
        xlabel="Number of transactions in weeks 1-39",
        ylabel="Average number of transactions in weeks 40-78",
        title="Figure 3 — Conditional Expectation of Future Transactions for CDNOW"
    )
    ax.legend(frameon=False)
    sns.despine(fig)
    return fig

figure3_conditional(hb_m1, pnbd, txn)

# %% ------------------------- [Figure 4 — Scatter λ vs μ]
lam, mu = hb_m1.posterior_means()
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(lam, mu, s=12, alpha=0.5, edgecolor="none")
ax.set(xlabel=r"$\lambda$", ylabel=r"$\mu$",
       title="Figure 4 — Posterior Means of λ and μ")
sns.despine(fig)

# %% ------------------------- [Figure 5 — Histogram corr(logλ, logμ)]
draws = np.stack(hb_m1.draws["theta"], axis=0)      # (n_draws, N, 2)
corrs = [np.corrcoef(d[:, 0], d[:, 1])[0, 1] for d in draws]
sns.histplot(corrs, bins=30)
plt.title("Figure 5 — Posterior distribution of corr( log λᵢ , log μᵢ )")
plt.xlabel("Correlation"); plt.ylabel("Count")

# %% ------------------------- [Table 1 — Descriptive statistics]
raw = pd.read_csv(
    "~/.cache/cdnow_master.txt", sep=r"\s+", header=0,
    names=["id", "date", "qty", "spend"], parse_dates=["date"]
)
end_date = pd.Timestamp("1998-06-30")
first = raw.groupby("id")["date"].min()
last  = raw.groupby("id")["date"].max()
n_txn = raw.groupby("id").size()
init_amt = raw.sort_values("date").groupby("id")["spend"].first()

table1 = pd.DataFrame({
    "Number of repeats": n_txn - 1,
    "Observation duration T (days)": (end_date - first).dt.days,
    "Recency (T-t) (days)": (end_date - last).dt.days,
    "Amount of initial purchase ($)": init_amt,
}).agg(["mean", "std", "min", "max"]).T.round(2)
table1.columns = ["Mean", "Std. dev.", "Min", "Max"]
print("\nTable 1 — Descriptive Statistics for CDNOW Data")
print(tabulate(table1, headers="keys", tablefmt="github"))

# %% ------------------------- [Table 2 — Model fit]
def indiv_metrics(exp_val, act_val, exp_cal, act_cal):
    c_val = np.corrcoef(exp_val, act_val)[0, 1]
    c_cal = np.corrcoef(exp_cal, act_cal)[0, 1]
    mse_v = np.mean((exp_val - act_val) ** 2)
    mse_c = np.mean((exp_cal - act_cal) ** 2)
    return c_val, c_cal, mse_v, mse_c

act_cal = (
    txn[txn["t_week"] <= 39].groupby("id").size()
    .reindex(summary.index, fill_value=0).values
)
act_val = (
    txn[(txn["t_week"] > 39) & (txn["t_week"] <= 52)].groupby("id").size()
    .reindex(summary.index, fill_value=0).values
)

exp_cal_pn = pnbd.conditional_expected_number_of_purchases_up_to_time(39*7, freq, reci*7, Tarr*7)
exp52 = pnbd.conditional_expected_number_of_purchases_up_to_time(52*7, freq, reci*7, Tarr*7)
exp_val_pn = exp52 - exp_cal_pn

exp_cal_m1 = hb_expected(hb_m1, 39)[:, 0]
exp_val_m1 = hb_expected(hb_m1, 13)[:, 0]

exp_cal_m2 = hb_expected(hb_m2, 39)[:, 0]
exp_val_m2 = hb_expected(hb_m2, 13)[:, 0]

c_v_pn, c_c_pn, mse_v_pn, mse_c_pn = indiv_metrics(exp_val_pn, act_val, exp_cal_pn, act_cal)
c_v_m1, c_c_m1, mse_v_m1, mse_c_m1 = indiv_metrics(exp_val_m1, act_val, exp_cal_m1, act_cal)
c_v_m2, c_c_m2, mse_v_m2, mse_c_m2 = indiv_metrics(exp_val_m2, act_val, exp_cal_m2, act_cal)

weeks = np.arange(0, 53)
actual_cum = (
    txn[txn["t_week"] <= 52].groupby("t_week").size()
    .reindex(weeks, fill_value=0)
).cumsum()

def cum_curve_pnbd():
    return np.array([pnbd.conditional_expected_number_of_purchases_up_to_time(
                        w*7, freq, reci*7, Tarr*7).sum() for w in weeks])

def cum_curve_hb(hb_model):
    lam, mu = hb_model.posterior_means()
    return ((lam/mu)[:,None]*(1-np.exp(-mu[:,None]*weeks))).sum(axis=0)

cum_pn  = cum_curve_pnbd()
cum_m1  = cum_curve_hb(hb_m1)
cum_m2  = cum_curve_hb(hb_m2)

def mape(curve, start, end):
    slice_act = actual_cum[start:end]
    slice_pred = curve[start:end]
    return np.mean(np.abs(slice_pred - slice_act) / slice_act.replace(0, np.nan)) * 100

mape_val_pn, mape_cal_pn = mape(cum_pn, 40, None), mape(cum_pn, 0, 40)
mape_val_m1, mape_cal_m1 = mape(cum_m1, 40, None), mape(cum_m1, 0, 40)
mape_val_m2, mape_cal_m2 = mape(cum_m2, 40, None), mape(cum_m2, 0, 40)

mape_pool_pn = mape(cum_pn, 0, None)
mape_pool_m1 = mape(cum_m1, 0, None)
mape_pool_m2 = mape(cum_m2, 0, None)

table2 = pd.DataFrame({
    "Criterion": [
        "Correlation: Validation", "Correlation: Calibration",
        "MSE: Validation", "MSE: Calibration",
        "MAPE (%): Validation", "MAPE (%): Calibration", "MAPE (%): Pooled"
    ],
    "Pareto/NBD": [
        c_v_pn, c_c_pn, mse_v_pn, mse_c_pn,
        mape_val_pn, mape_cal_pn, mape_pool_pn
    ],
    "HB M1 (no cov)": [
        c_v_m1, c_c_m1, mse_v_m1, mse_c_m1,
        mape_val_m1, mape_cal_m1, mape_pool_m1
    ],
    "HB M2 (with cov)": [
        c_v_m2, c_c_m2, mse_v_m2, mse_c_m2,
        mape_val_m2, mape_cal_m2, mape_pool_m2
    ],
}).set_index("Criterion").round(2)

print("\nTable 2 — Model Fit Statistics")
print(tabulate(table2, headers="keys", tablefmt="github"))


# %% ------------------------- [Table 3 — Estimation Results for CDNOW Data]
def table3_estimation(hb_model, label):
    """Return a dict of posterior mean and 95% HPD for β and Γ."""
    beta_draws   = np.stack(hb_model.draws["beta"], axis=0)     # (n_draws,K,2)
    Gamma_draws  = np.stack(hb_model.draws["Gamma"], axis=0)    # (n_draws,2,2)

    stats = {}
    # ----- β coefficients -----------------------------------------
    stats[f"{label} β0_λ"] = np.percentile(beta_draws[:,0,0], [50, 2.5, 97.5])
    stats[f"{label} β0_μ"] = np.percentile(beta_draws[:,0,1], [50, 2.5, 97.5])
    if beta_draws.shape[1] > 1:
        stats[f"{label} β1_λ"] = np.percentile(beta_draws[:,1,0], [50, 2.5, 97.5])
        stats[f"{label} β1_μ"] = np.percentile(beta_draws[:,1,1], [50, 2.5, 97.5])

    # ----- Γ elements --------------------------------------------
    g00 = Gamma_draws[:,0,0]; g11 = Gamma_draws[:,1,1]; g01 = Gamma_draws[:,0,1]
    stats[f"{label} var_logλ"] = np.percentile(g00, [50, 2.5, 97.5])
    stats[f"{label} var_logμ"] = np.percentile(g11, [50, 2.5, 97.5])
    stats[f"{label} cov"]      = np.percentile(g01, [50, 2.5, 97.5])
    corr = g01 / np.sqrt(g00 * g11)
    stats[f"{label} corr"]     = np.percentile(corr, [50, 2.5, 97.5])

    # marginal log‑likelihood placeholder (NA)
    stats[f"{label} loglike"]  = [np.nan, np.nan, np.nan]
    return stats

m1_stats = table3_estimation(hb_m1, "M1")
m2_stats = table3_estimation(hb_m2, "M2")

rows = [
    ("Intercept βλ",           m1_stats["M1 β0_λ"], m2_stats["M2 β0_λ"]),
    ("Intercept βμ",           m1_stats["M1 β0_μ"], m2_stats["M2 β0_μ"]),
    ("Initial amt βλ",         ["—","—","—"]       , m2_stats["M2 β1_λ"]),
    ("Initial amt βμ",         ["—","—","—"]       , m2_stats["M2 β1_μ"]),
    ("var[log λ]",             m1_stats["M1 var_logλ"], m2_stats["M2 var_logλ"]),
    ("var[log μ]",             m1_stats["M1 var_logμ"], m2_stats["M2 var_logμ"]),
    ("cov[log λ,log μ]",       m1_stats["M1 cov"],      m2_stats["M2 cov"]),
    ("Correlation Γ₀",         m1_stats["M1 corr"],     m2_stats["M2 corr"]),
    ("Marginal log-likelihood", m1_stats["M1 loglike"], m2_stats["M2 loglike"]),
]

def _fmt(t):
    """
    Format (median, lower, upper) as 'm\n(l,u)'.
    Returns '—' if t is not numeric.
    """
    try:
        m, lo, hi = float(t[0]), float(t[1]), float(t[2])
        if np.isfinite(m):
            return f"{m:.2f}\n({lo:.2f},{hi:.2f})"
    except (TypeError, ValueError, IndexError):
        pass
    return "—"

tbl3 = pd.DataFrame(
    {
        "HB M1 (no covariate)"  : [_fmt(v)  for _, v, _ in rows],
        "HB M2 (with covariate)": [_fmt(v2) for _, _, v2 in rows],
    },
    index=[r[0] for r in rows]
)
print("\nTable 3 — Estimation Results for CDNOW Data")
print(tabulate(tbl3, headers="keys", tablefmt="github"))


# %% ------------------------- [Table 4 — Customer‑Specific top/bottom 10]
def customer_stats(hb_model):
    thetas = np.stack(hb_model.draws["theta"], axis=0)     # (n_draws,N,2)
    lam_draws = np.exp(thetas[:,:,0])
    mu_draws  = np.exp(thetas[:,:,1])

    mean_lam = lam_draws.mean(axis=0)
    q025_lam = np.quantile(lam_draws, 0.025, axis=0)
    q975_lam = np.quantile(lam_draws, 0.975, axis=0)

    mean_mu  = mu_draws.mean(axis=0)
    q025_mu  = np.quantile(mu_draws, 0.025, axis=0)
    q975_mu  = np.quantile(mu_draws, 0.975, axis=0)

    # expected lifetime in years (weeks -> years)
    mean_life = (1 / mean_mu) / 52

    # probability alive at end of calibration
    T_weeks = Tarr
    p_alive = 1 / (1 + (mean_mu / (mean_lam + mean_mu)) * (np.exp((mean_lam + mean_mu)*(T_weeks - summary["recency"].values)) - 1))

    # expected validation repeats (13 weeks)
    exp_val = mean_lam/mean_mu * (1 - np.exp(-mean_mu*13))

    df = pd.DataFrame({
        "ID"   : summary.index,
        "Mean(λ)"  : mean_lam, "λ 2.5%" : q025_lam, "λ 97.5%" : q975_lam,
        "Mean(μ)"  : mean_mu,  "μ 2.5%" : q025_mu,  "μ 97.5%" : q975_mu,
        "Lifetime(yrs)" : mean_life,
        "P(alive)"     : p_alive,
        "Exp val txns" : exp_val,
    }).set_index("ID")
    return df

df_cust = customer_stats(hb_m1)

top10 = df_cust.nlargest(10, "Mean(λ)")
bot10 = df_cust.nsmallest(10, "Mean(λ)")

table4 = pd.concat([top10, bot10]).round(3)
print("\nTable 4 — Customer‑Specific Statistics (Top & Bottom 10 by λ)")
print(tabulate(table4, headers="keys", tablefmt="github"))
# %%
