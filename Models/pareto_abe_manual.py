"""pareto_abe.py
Python re-implementation of the hierarchical-Bayes Pareto/NBD variant from
Abe (2009):

* Synthetic-data generator identical to `abe.GenerateData` from the original R
  code.
* Event-log → CBS helper (`elog2cbs`).
* Vectorised latent-variable samplers (`draw_z`, `draw_tau`).
* Full Metropolis-within-Gibbs sampler (`mcmc_draw_parameters`) that mirrors the
  algorithm in Abe's Appendix A.1. No fancy back-end - NumPy + SciPy are fast
  enough; PyMC is **optional** (only used for trace plotting later).

The output roughly matches the R function: two dictionaries with draws for each
chain.  Results are easy to stuff into an ArviZ `InferenceData` if desired.

All times are measured in **weeks** (as in the R demo). The random-number flow
is managed by a `numpy.random.Generator` passed around explicitly for
reproducibility.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports & typing
# -----------------------------------------------------------------------------
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import invwishart

try:
    import arviz as az  # optional – only used when the user asks for diagnostics
except ImportError:  # pragma: no cover – az is optional
    az = None  # type: ignore

__all__ = [
    # data helpers
    "CustomerCBS",
    "elog2cbs",
    "generate_pareto_abe",
    # latent draws
    "draw_z",
    "draw_tau",
    # Gibbs sampler
    "mcmc_draw_parameters",
    "draw_future_transactions",
]

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class CustomerCBS:
    """Sufficient statistics for one customer in the calibration window."""

    x: int        # repeat transactions (excl. first purchase)
    t_x: float    # recency (weeks since first purchase)
    T_cal: float  # calibration period length (weeks)

    @property
    def frequency(self) -> int:
        return self.x

    @property
    def recency(self) -> float:
        return self.t_x

# -----------------------------------------------------------------------------
# Helper – convert event log to CBS
# -----------------------------------------------------------------------------

def elog2cbs(elog: pd.DataFrame, T_cal: float) -> pd.DataFrame:
    """Create RFM‑style summary statistics from an event log.

    Parameters
    ----------
    elog : DataFrame with columns ``cust`` and ``t`` (weeks since cohort start).
    T_cal : float, calibration‑window end (same unit as ``t``).
    """
    grouped = elog[elog["t"] <= T_cal].groupby("cust")
    out = grouped["t"].agg([("x", "count"), ("t_x", "max")]).reset_index()

    # In Abe’s notation x excludes the first purchase (always at t=0)
    out["x"] = np.clip(out["x"] - 1, 0, None)
    out["T_cal"] = T_cal
    return out

# -----------------------------------------------------------------------------
# Data generator – Abe (2009) exact replica
# -----------------------------------------------------------------------------

def generate_pareto_abe(
    n: int,
    T_cal: float | np.ndarray,
    T_star: float | np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    covars: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate customer transactions under Abe (2009).

    Returns
    -------
    cbs : calibration‑period summary statistics plus true parameters
    elog : full event log (cust, t, date is omitted – just weeks)
    """
    rng = np.random.default_rng(seed)

    beta = np.asarray(beta, dtype=float)
    K, D = beta.shape
    assert D == 2, "beta must have two columns (log‑lambda, log‑mu)"

    # --- covariates X_i ------------------------------------------------------
    if covars is None:
        covars = np.column_stack([
            np.ones(n),
            rng.uniform(-1.0, 1.0, size=(n, K - 1)),
        ])
    else:
        covars = np.asarray(covars, dtype=float)
        if covars.ndim == 1:
            covars = covars[:, None]
        if not np.allclose(covars[:, 0], 1):
            covars = np.column_stack([np.ones(covars.shape[0]), covars])
        if covars.shape != (n, K):
            raise ValueError("covars has wrong shape relative to beta")

    # --- individual "true" parameters --------------------------------------
    theta = np.exp(
        covars @ beta + rng.multivariate_normal(np.zeros(2), gamma, size=n)
    )  # (n, 2)
    lambdas, mus = theta[:, 0], theta[:, 1]
    taus = rng.exponential(scale=1.0 / mus)

    # --- timing --------------------------------------------------------------
    T_cal = np.asarray(T_cal, dtype=float).ravel()
    if T_cal.size == 1:
        T_cal = np.full(n, T_cal.item())
    T_cal_fix = T_cal.max()

    T_star = np.asarray(T_star, dtype=float).ravel()
    T_zero = T_cal_fix - T_cal  # birth offsets

    # --- simulate event log --------------------------------------------------
    elog_rows: list[tuple[int, float]] = []
    for i in range(n):
        lam, tau = float(lambdas[i]), float(taus[i])
        min_T = min(float(T_cal[i] + T_star.max()), tau)
        # accumulate inter‑purchase times until coverage ≥ min_T
        t_acc, ts = 0.0, [0.0]
        while t_acc < min_T:
            dt = rng.exponential(scale=1.0 / lam)
            t_acc += dt
            ts.append(t_acc)
        ts = np.array(ts)
        ts = ts[ts <= tau] + T_zero[i]  # alive + birth shift
        ts = ts[ts <= T_cal_fix + T_star.max()]
        elog_rows.extend([(i + 1, float(t)) for t in ts])

    elog = pd.DataFrame(elog_rows, columns=["cust", "t"], dtype=float)

    # --- CBS summary ---------------------------------------------------------
    cbs = elog2cbs(elog, float(T_cal_fix))
    cbs["lambda_true"], cbs["mu_true"], cbs["tau_true"] = lambdas, mus, taus
    cbs["alive_true"] = (T_zero + taus) > T_cal_fix

    # hold‑out transactions for each requested horizon
    for t_star in T_star:
        col = f"x_star{int(t_star)}" if T_star.size > 1 else "x_star"
        upper = T_cal_fix + t_star
        cnts = (
            elog[(elog["t"] > T_cal_fix) & (elog["t"] <= upper)]
            .groupby("cust")["t"].size()
            .reindex(cbs["cust"], fill_value=0)
            .values
        )
        cbs[col] = cnts

    # copy covariates for inspection
    for j in range(K):
        cbs[f"cov{j}"] = covars[:, j]

    return cbs, elog

# -----------------------------------------------------------------------------
# Latent‑variable samplers – NumPy translations of draw_z / draw_tau
# -----------------------------------------------------------------------------

def draw_z(cbs: pd.DataFrame, lambdas: np.ndarray, mus: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    tx, Tcal = cbs["t_x"].to_numpy(), cbs["T_cal"].to_numpy()
    mu_lam = mus + lambdas
    p_alive = 1.0 / (1.0 + (mus / mu_lam) * (np.exp(mu_lam * (Tcal - tx)) - 1.0))
    return rng.random(p_alive.shape) < p_alive  # boolean array


def draw_tau(
    cbs: pd.DataFrame,
    lambdas: np.ndarray,
    mus: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    tx, Tcal = cbs["t_x"].to_numpy(), cbs["T_cal"].to_numpy()
    mu_lam = mus + lambdas
    tau = np.empty_like(tx)

    # alive – left‑truncated exponential
    alive_idx = np.where(z)[0]
    if alive_idx.size:
        tau[alive_idx] = Tcal[alive_idx] + rng.exponential(scale=1.0 / mus[alive_idx])

    # churned – double‑truncated exponential over [t_x, T_cal]
    churn_idx = np.where(~z)[0]
    if churn_idx.size:
        ml = mu_lam[churn_idx]
        ml_tx = np.minimum(700.0, ml * tx[churn_idx])
        ml_T = np.minimum(700.0, ml * Tcal[churn_idx])
        u = rng.random(churn_idx.size)
        tau[churn_idx] = -np.log((1 - u) * np.exp(-ml_tx) + u * np.exp(-ml_T)) / ml
    return tau

# -----------------------------------------------------------------------------
# Conjugate update for (β, Σ)  –  replaces bayesm::rmultireg
# -----------------------------------------------------------------------------

def _draw_level_2(
    X: np.ndarray,  # (N, K)
    log_lambda: np.ndarray,
    log_mu: np.ndarray,
    hyper: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior draw for β (K×2) and Σ (2×2) in a normal regression."""
    Y = np.column_stack([log_lambda, log_mu])  # (N, 2)

    A0 = hyper["A_0"]  # (K, K) precision
    B0 = hyper["beta_0"]  # (K, 2)
    nu0 = hyper["nu_00"]
    S0 = hyper["gamma_00"]  # (2, 2)

    XtX = X.T @ X
    V_beta = np.linalg.inv(XtX + A0)  # (K, K)
    B_hat = V_beta @ (X.T @ Y + A0 @ B0)  # (K, 2)

    # scale matrix for Inv‑Wishart
    E = Y - X @ B_hat
    C = B_hat - B0
    S_n = S0 + E.T @ E + C.T @ A0 @ C
    nu_n = nu0 + X.shape[0]

    Sigma = invwishart.rvs(df=nu_n, scale=S_n, random_state=rng)

    # β | Σ ~ MN(B_hat, Σ, V_beta)
    beta = rng.multivariate_normal(B_hat.ravel(), np.kron(Sigma, V_beta)).reshape(B_hat.shape)
    return beta, Sigma

# -----------------------------------------------------------------------------
# MH step for (λ_i, μ_i)
# -----------------------------------------------------------------------------

def _draw_level_1(
    cbs: pd.DataFrame,
    X: np.ndarray,
    lambdas: np.ndarray,
    mus: np.ndarray,
    z: np.ndarray,
    tau: np.ndarray,
    beta: np.ndarray,
    Sigma: np.ndarray,
    rng: np.random.Generator,
    n_mh_steps: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    x = cbs["x"].to_numpy()
    T_cal = cbs["T_cal"].to_numpy()

    inv_Sigma = np.linalg.inv(Sigma)
    mv_mean = X @ beta  # (N, 2)

    log_lambda = np.log(lambdas)
    log_mu = np.log(mus)

    N = log_lambda.size

    def log_posterior(ll, lm):
        """Vectorised log‑posterior for each customer."""
        diff_l = ll - mv_mean[:, 0]
        diff_m = lm - mv_mean[:, 1]
        lik = (
            x * ll
            + (1 - z) * lm
            - (np.exp(ll) + np.exp(lm)) * (z * T_cal + (1 - z) * tau)
        )
        prior = (
            -0.5
            * (
                diff_l ** 2 * inv_Sigma[0, 0]
                + 2 * diff_l * diff_m * inv_Sigma[0, 1]
                + diff_m ** 2 * inv_Sigma[1, 1]
            )
        )
        res = lik + prior
        res = np.where(lm > 5.0, -np.inf, res)  # cap
        return res

    cur_lp = log_posterior(log_lambda, log_mu)

    for _ in range(n_mh_steps):
        # draw two independent t3-noise vectors of length N
        eps_lambda = Sigma[0, 0] * rng.standard_t(df=3, size=N)
        eps_mu     = Sigma[1, 1] * rng.standard_t(df=3, size=N)
        prop_ll = log_lambda + eps_lambda
        prop_lm = log_mu     + eps_mu                         #  ← NEW loop

        # 4) cap on [-70, 70]
        # (this is a hack to avoid numerical issues with exp() in the posterior)
        prop_ll = np.clip(prop_ll, -70.0, 70.0)
        prop_lm = np.clip(prop_lm, -70.0, 70.0)

        # 5) Compute Metropolis‑Hastings ratio
        # (log posterior for proposal minus log posterior for current)
        prop_lp = log_posterior(prop_ll, prop_lm)
        mhr     = np.exp(prop_lp - cur_lp)
        accept  = mhr > rng.random(cur_lp.shape)

        # 6) Update current values
        log_lambda[accept] = prop_ll[accept]
        log_mu[accept]     = prop_lm[accept]
        cur_lp[accept]     = prop_lp[accept]

    lambdas = np.exp(log_lambda)
    mus = np.exp(log_mu)
    return lambdas, mus


# -----------------------------------------------------------------------------
# Main Gibbs driver (single chain) – mirrors abe.mcmc.DrawParameters
# -----------------------------------------------------------------------------

def _run_chain(
    chain_id: int,
    cbs: pd.DataFrame,
    X: np.ndarray,
    hyper: Dict[str, Any],
    mcmc: int,
    burnin: int,
    thin: int,
    rng: np.random.Generator,
    trace: int,
    n_mh_steps: int, 
) -> Dict[str, Any]:
    N = cbs.shape[0]
    K = X.shape[1]
    n_draws = (mcmc - 1) // thin + 1

    # storage
    lvl1_draws = np.empty((n_draws, N, 4), dtype=float)  # λ μ τ z
    lvl2_draws = np.empty((n_draws, 2 * K + 3), dtype=float)
    log_likelihood_chain = []

    # initialise λ & μ with simple heuristics --------------------------------
    lam_init = cbs["x"].mean() / np.mean(np.where(cbs["t_x"] == 0, cbs["T_cal"], cbs["t_x"]))
    lambdas = np.full(N, lam_init)
    mus = 1.0 / (cbs["t_x"].to_numpy() + 0.5 / lam_init)

    # update intercept of β₀ to empirical mean (matches R code)
    hyper["beta_0"][0, 0] = math.log(lambdas.mean())
    hyper["beta_0"][0, 1] = math.log(mus.mean())

    # dummy placeholders – will be overwritten in first iteration
    z = np.ones(N, dtype=bool)
    tau = cbs["T_cal"].to_numpy() + 1.0
    beta, Sigma = hyper["beta_0"], hyper["gamma_00"].copy()

    store_idx = -1
    tot_steps = burnin + mcmc
    for step in range(1, tot_steps + 1):
        if trace and step % trace == 0:
            print(f"chain {chain_id} | step {step}/{tot_steps}")

        # 1) z, τ conditional on current λ, μ
        z = draw_z(cbs, lambdas, mus, rng)

        tau = draw_tau(cbs, lambdas, mus, z, rng)

        # 2) β, Σ  (level‑2 hyper‑parameters)
        beta, Sigma = _draw_level_2(X, np.log(lambdas), np.log(mus), hyper, rng)

        # 3) λ, μ (level‑1) via MH
        lambdas, mus = _draw_level_1(
            cbs, X, lambdas, mus, z, tau, beta, Sigma, rng,
            n_mh_steps=n_mh_steps,
        )

        # store draws after burn‑in / thinning
        if step > burnin and (step - 1 - burnin) % thin == 0:
            store_idx += 1
            # Correction safeguard: ensure lambdas and mus are in natural scale
            lambdas = np.exp(np.log(lambdas))
            mus = np.exp(np.log(mus))
            lvl1_draws[store_idx, :, 0] = lambdas
            lvl1_draws[store_idx, :, 1] = mus
            lvl1_draws[store_idx, :, 2] = tau
            lvl1_draws[store_idx, :, 3] = z.astype(float)
            lvl2_draws[store_idx, : 2 * K] = beta.T.ravel()
            lvl2_draws[store_idx, -3:] = [Sigma[0, 0], Sigma[0, 1], Sigma[1, 1]]
            # Compute log-likelihood for this draw and store
            # Use the log-posterior function from _draw_level_1
            x = cbs["x"].to_numpy()
            T_cal = cbs["T_cal"].to_numpy()
            mv_mean = X @ beta  # (N, 2)
            log_lambda = np.log(lambdas)
            log_mu = np.log(mus)
            # log-likelihood (excluding prior terms)
            # For each customer, the likelihood part is:
            # lik = x * log_lambda + (1 - z) * log_mu - (lambdas + mus) * (z * T_cal + (1 - z) * tau)
            lik = (
                x * log_lambda
                + (1 - z) * log_mu
                - (lambdas + mus) * (z * T_cal + (1 - z) * tau)
            )
            log_likelihood_chain.append(np.mean(lik))

    log_likelihood_chain = np.array(log_likelihood_chain)
    return dict(level_1=lvl1_draws, level_2=lvl2_draws, log_likelihood=log_likelihood_chain)

# -----------------------------------------------------------------------------
# Public API – run multiple chains
# -----------------------------------------------------------------------------

def mcmc_draw_parameters(
    cal_cbs: pd.DataFrame,
    covariates: Sequence[str] | None = None,
    mcmc: int = 2500,
    burnin: int = 500,
    thin: int = 50,
    chains: int = 2,
    seed: Optional[int] = None,
    trace: int = 100,
    n_mh_steps: int = 20,          #  ← NEW default
) -> Dict[str, Any]:
    """Run Abe (2009) Gibbs sampler on calibration CBS.

    Parameters
    ----------
    cal_cbs : DataFrame with columns ["x", "t_x", "T_cal"] + optional covars.
    covariates : list of column names to include (defaults to none ⇒ intercept only).
    Returns
    -------
    dict with keys ``level_1`` and ``level_2`` – each a list with one entry per
    chain, containing the raw NumPy draws.
    """
    if covariates is None:
        covariates = []
    for col in ("x", "t_x", "T_cal"):
        if col not in cal_cbs:
            raise ValueError(f"cal_cbs missing required column '{col}'")
    if not all(col in cal_cbs for col in covariates):
        raise ValueError("some covariate columns not in cal_cbs")

    cbs = cal_cbs.copy().reset_index(drop=True)
    cbs["intercept"] = 1.0
    cols = ["intercept"] + list(covariates)
    X = cbs[cols].to_numpy(float)

    K = X.shape[1]
    # diffuse NIW hyper‑priors (matches bayesm defaults)
    beta_0 = np.zeros((K, 2))
    A_0 = np.eye(K) * 0.01  # small precision ⇒ vague prior
    nu_00 = 3 + K
    gamma_00 = nu_00 * np.eye(2)

    hyper = dict(beta_0=beta_0, A_0=A_0, nu_00=nu_00, gamma_00=gamma_00)

    # run chains sequentially (easy to parallelise later)
    all_lvl1: list[np.ndarray] = []
    all_lvl2: list[np.ndarray] = []
    log_liks: list[np.ndarray] = []
    for ch in range(chains):
        rng = np.random.default_rng(None if seed is None else seed + ch)
        draws = _run_chain(
            ch + 1,
            cbs,
            X,
            {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in hyper.items()},
            mcmc,
            burnin,
            thin,
            rng,
            trace,
            n_mh_steps, 
        )
        all_lvl1.append(draws["level_1"])
        all_lvl2.append(draws["level_2"])
        log_liks.append(draws["log_likelihood"])

    marginal_loglik = np.mean(np.concatenate(log_liks))
    return dict(level_1=all_lvl1, level_2=all_lvl2, log_likelihood=marginal_loglik)

def draw_future_transactions(cbs: pd.DataFrame, draws: Dict[str, Any], T_star: float = 39.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate future transactions for each customer given posterior draws.
    
    Parameters
    ----------
    cbs : pd.DataFrame
        Customer summary statistics with columns ['T_cal'].
    draws : dict
        Output of `mcmc_draw_parameters`, containing level_1 samples.
    T_star : float
        Forecast horizon (same time unit as T_cal).
    seed : int, optional
        Seed for reproducible random draws.

    Returns
    -------
    np.ndarray
        Simulated x_star values, shape (n_draws, n_customers)
    """
    rng = np.random.default_rng(seed)
    lvl1_all_chains = draws["level_1"]
    x_stars = []

    for chain in lvl1_all_chains:
        for draw in chain:  # shape: (n_customers, 4)
            lambdas = draw[:, 0]
            mus = draw[:, 1]
            tau = draw[:, 2]
            z = draw[:, 3] > 0.5  # Convert to boolean

            T_cal = cbs["T_cal"].to_numpy()
            tau_star = np.full_like(T_cal, T_star, dtype=float)
            # For churned customers, tau_star is min(tau - T_cal, T_star)
            tau_star[~z] = np.clip(tau[~z] - T_cal[~z], 0.0, T_star)

            # Poisson sampling for all customers using lam=lambdas * tau_star
            x_draws = rng.poisson(lam=lambdas * tau_star)
            x_stars.append(x_draws)

    return np.array(x_stars)  # shape: (n_draws, n_customers)


# -----------------------------------------------------------------------------
# Quick smoke‑test – run tiny MCMC on generated data when executed as script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    beta = np.array([[0.18, -2.5]])  # intercept only
    gamma = np.array([[0.05, 0.1], [0.1, 0.2]])

    cbs, _ = generate_pareto_abe(50, 32, 32, beta, gamma, seed=42)
    draws = mcmc_draw_parameters(cbs, mcmc=100, burnin=50, thin=10, chains=1, trace=20, seed=123)
    print("level_2 draws shape:", draws["level_2"][0].shape)

