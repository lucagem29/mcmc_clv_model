import numpy as np
import pandas as pd
from scipy.special import gammaln # for log‑factorial constant
# ---------------------------------------------------------------------

def summarize_level2(draws_level2: np.ndarray, param_names: list[str], decimals: int = 2) -> pd.DataFrame:
    """
    Summarize level-2 MCMC draws into a DataFrame of 2.5%, 50%, and 97.5% quantiles
    for each model parameter.
    """
    quantiles = np.percentile(draws_level2, [2.5, 50, 97.5], axis=0)
    summary = pd.DataFrame(quantiles.T, columns=["2.5%", "50%", "97.5%"], index=param_names)
    return summary.round(decimals)

def post_mean_lambdas(draws):
    """
    Compute the per-customer posterior mean of the purchase-rate (λ) from level-1 draws.
    """
    all_draws = np.concatenate(draws["level_1"], axis=0)
    return all_draws[:, :, 0].mean(axis=0)

def post_mean_mus(draws):
    """
    Compute the per-customer posterior mean of the dropout hazard (μ) from level-1 draws.
    """
    all_draws = np.concatenate(draws["level_1"], axis=0)
    return all_draws[:, :, 1].mean(axis=0)

def mape_aggregate(actual, pred):
    """
    Calculate the time-series Mean Absolute Percentage Error (MAPE) as defined in Abe (2009):
    the average absolute cumulative deviation divided by the final cumulative actual.
    """
    cum_a = np.cumsum(actual)
    cum_p = np.cumsum(pred)
    abs_error = np.abs(cum_p - cum_a)
    return abs_error.mean() / cum_a[-1] * 100

def extract_correlation(draws_level2):
    """
    Compute 2.5%, 50%, and 97.5% quantiles of the correlation between log(λ) and log(μ)
    from level-2 covariance and variance draws.
    """
    cov = draws_level2[:, -2]  # cov_log_lambda_mu
    var_lambda = draws_level2[:, -3]
    var_mu = draws_level2[:, -1]
    corr = cov / np.sqrt(var_lambda * var_mu)
    return np.percentile(corr, [2.5, 50, 97.5]).round(2)


# --- compute chain‑averaged log‑likelihood ---------------------------------
def chain_total_loglik(level1_chains, cbs):
    """
    Compute the average total log-likelihood over all level-1 MCMC draws for each customer.
    """
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


def compute_table4(draws, xstar_draws):
    """
    Generate Table 4: customer-level RFM statistics including top/bottom segments and summary
    rows, matching the format in Abe (2009).
    """
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
