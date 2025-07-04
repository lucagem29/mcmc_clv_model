"""pareto_abe.py
Python re-implementation of the hierarchical-Bayes Pareto/NBD variant from
Abe (2009):

* Synthetic-data generator identical to `abe.GenerateData` from the original R
  code.
* Event-log → CBS helper (`elog2cbs`).
* Vectorised latent-variable samplers (`draw_z`, `draw_tau`).
* Full Metropolis-within-Gibbs sampler (`mcmc_draw_parameters_rfm_m`) that mirrors the
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
from src.models.utils.elog2cbs2param import elog2cbs

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
    "mcmc_draw_parameters_rfm_m",
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
    exp_neg_z = np.exp(-mu_lam * (Tcal - tx))
    p_alive   = (mu_lam * exp_neg_z) / (mu_lam * exp_neg_z + mus * (1.0 - exp_neg_z)) 
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

def draw_eta(
    cbs: pd.DataFrame,
    log_eta: np.ndarray,
    beta: np.ndarray,
    Sigma: np.ndarray,
    omega2: float,
    rng: np.random.Generator,
    covariate_cols: List[str],
) -> np.ndarray:
    """
    Conjugate Normal draw for η_i (log-average spend).
    Data: log_s_i ~ N(η_i, ω^2)
    Prior: η_i ~ MVN((X @ beta)_3, Sigma[2,2])
    """
    # 1) Compute prior mean and prior variance for η
    prior_mean = (cbs[covariate_cols].to_numpy() @ beta)[:, 2]
    prior_var  = Sigma[2, 2]

    # 2) Posterior precision and variance
    post_precision = 1.0/omega2 + 1.0/prior_var
    post_var       = 1.0/post_precision

    # 3) Posterior mean
    log_s = cbs["log_s"].to_numpy()
    post_mean = post_var * (log_s/omega2 + prior_mean/prior_var)

    # 4) Sample η ~ Normal(post_mean, sqrt(post_var))
    return rng.normal(post_mean, np.sqrt(post_var))


# -----------------------------------------------------------------------------
# Conjugate update for (β, Σ)  –  replaces bayesm::rmultireg
# -----------------------------------------------------------------------------

def _draw_level_2(
    X: np.ndarray,            # (N, K)
    log_lambda: np.ndarray,   # (N,)
    log_mu: np.ndarray,       # (N,)
    log_eta: np.ndarray,      # (N,)
    hyper: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Posterior draw for β (K×3) and Σ (3×3) in a multivariate normal regression.
    Now handles θ_i = (logλ_i, logμ_i, logη_i).
    """
    # 1) Stack the three responses
    Y = np.column_stack([log_lambda, log_mu, log_eta])  # shape (N, 3)

    # 2) Pull in hyper‐priors
    A0    = hyper["A_0"]      # prior precision (K×K)
    B0    = hyper["beta_0"]   # prior mean (K×3)
    nu0   = hyper["nu_00"]    # prior degrees of freedom
    S0    = hyper["gamma_00"] # prior scale matrix (3×3)

    # 3) Compute posterior for β | Σ
    XtX    = X.T @ X                   # (K×K)
    V_beta = np.linalg.inv(XtX + A0)   # (K×K)
    B_hat  = V_beta @ (X.T @ Y + A0 @ B0)  # (K×3)

    # 4) Compute updated scale for Σ posterior
    E   = Y - X @ B_hat                # residuals (N×3)
    C   = B_hat - B0                   # deviation from prior mean (K×3)
    S_n = S0 + E.T @ E + C.T @ A0 @ C  # (3×3)
    nu_n = nu0 + X.shape[0]            # updated degrees of freedom

    # 5) Draw Σ ~ Inv-Wishart(nu_n, S_n)
    Sigma = invwishart.rvs(df=nu_n, scale=S_n, random_state=rng)

    # 6) Draw β ~ Matrix-Normal(B_hat, Σ, V_beta)
    cov = np.kron(Sigma, V_beta)       # (3K × 3K)
    beta_flat = rng.multivariate_normal(B_hat.ravel(), cov)
    beta = beta_flat.reshape(B_hat.shape)  # (K×3)

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
    covariate_cols: List[str], 
) -> Dict[str, Any]:
    N = cbs.shape[0]
    K = X.shape[1]
    n_draws = (mcmc - 1) // thin + 1

    # storage
    lvl1_draws = np.empty((n_draws, N, 5), dtype=float)      # λ, μ, τ, z, η
    lvl2_draws = np.empty((n_draws, 3*K + 6), dtype=float)   # β:K×3 + 6 unique entries of 3×3 covariance
    # log-likelihood chain for diagnostics
    log_likelihood_chain = []

    # initialise λ & μ with simple heuristics --------------------------------
    lam_init = cbs["x"].mean() / np.mean(np.where(cbs["t_x"] == 0, cbs["T_cal"], cbs["t_x"]))
    lambdas = np.full(N, lam_init)
    mus = 1.0 / (cbs["t_x"].to_numpy() + 0.5 / lam_init)
    # initialize η (log-spend) to zeros → eta in natural scale = 1
    eta = np.ones(N, dtype=float)
    omega2 = cbs["log_s"].var()  # variance of log-spend, used by draw_eta

    # update intercept of β₀ to empirical mean (matches R code)
    hyper["beta_0"][0, 0] = math.log(lambdas.mean())
    hyper["beta_0"][0, 1] = math.log(mus.mean())
    hyper["beta_0"][0, 2] = cbs["log_s"].mean()

    # dummy placeholders – will be overwritten in first iteration
    z = np.ones(N, dtype=bool)
    tau = cbs["T_cal"].to_numpy() + 1.0
    beta, Sigma = hyper["beta_0"], hyper["gamma_00"].copy()

    store_idx = -1
    tot_steps = burnin + mcmc
    for step in range(1, tot_steps + 1):
        if trace and step % trace == 0:
            print(f"chain {chain_id} | step {step}/{tot_steps}")

        # 1) draw z (alive indicator)
        z   = draw_z(cbs, lambdas, mus, rng)

        # 2) draw tau (dropout time)
        tau = draw_tau(cbs, lambdas, mus, z, rng)

        # 3) Metropolis–Hastings update for (λ, μ)
        lambdas, mus = _draw_level_1(
            cbs, X, lambdas, mus, z, tau, beta, Sigma, rng, n_mh_steps
        )

        # 4) clamp eta to avoid log(0), then draw η via conjugate Normal
        eta      = np.clip(eta, 1e-6, None)                    # prevents zeros
        log_eta  = np.log(eta)                                 # safe log
        eta      = np.exp(draw_eta(cbs, log_eta, beta, Sigma, omega2, rng, covariate_cols))

        # 5) update cohort-level β and Σ for θ = (logλ, logμ, logη)
        beta, Sigma = _draw_level_2(
            X,
            np.log(lambdas),
            np.log(mus),
            np.log(eta),
            hyper,
            rng
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
            lvl1_draws[store_idx, :, 4] = eta  # log-spend
            lvl1_draws[store_idx, :, 3] = z.astype(float)
            lvl2_draws[store_idx, :3 * K] = beta.T.ravel()
            lvl2_draws[store_idx, -6:]    = [
                Sigma[0,0], Sigma[0,1], Sigma[0,2],
                 Sigma[1,1], Sigma[1,2],
                              Sigma[2,2],
]

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

def mcmc_draw_parameters_rfm_m(
    cal_cbs: pd.DataFrame,
    covariates: Sequence[str] | None = None,
    mcmc: int = 2500,
    burnin: int = 500,
    thin: int = 50,
    chains: int = 2,
    seed: Optional[int] = None,
    trace: int = 100,
    n_mh_steps: int = 20,
) -> Dict[str, Any]:
    """
    3-dimensional HB Pareto/NBD + Spend (“RFM–M”) sampler.
    Returns posterior draws for θ_i = (log λ_i, log μ_i, log η_i)
    plus cohort-level β (K×3) and Σ (3×3).

    Parameters
    ----------
    cal_cbs : DataFrame
        Must contain columns ['cust','x','t_x','T_cal'] plus any covariates.
    covariates : list of str, optional
        Customer-level covariate columns (e.g. ['initial_spend_std']).
    mcmc, burnin, thin, chains, seed, trace, n_mh_steps
        MCMC settings.

    Returns
    -------
    dict with keys
      - level_1 : list of length `chains`, each an array (n_draws, N, 5) for [λ,μ,τ,z,η]
      - level_2 : list of length `chains`, each an array (n_draws, 3*K + 6)
      - log_likelihood : average log-likelihood across chains
    """
    if covariates is None:
        covariates = []
    # design matrix
    cbs = cal_cbs.copy().reset_index(drop=True)
    cbs["intercept"] = 1.0
    cols = ["intercept"] + list(covariates)
    X = cbs[cols].to_numpy(float)
    K = X.shape[1]

    # set up NIW priors for θ = (logλ, logμ, logη)
    beta_0   = np.zeros((K, 3))
    A_0      = np.eye(K) * 0.01
    nu_00    = 4 + K
    gamma_00 = nu_00 * np.eye(3)
    hyper    = dict(beta_0=beta_0, A_0=A_0, nu_00=nu_00, gamma_00=gamma_00)

    all_lvl1 = []
    all_lvl2 = []
    all_ll   = []

    for ch in range(chains):
        rng = np.random.default_rng(None if seed is None else seed + ch)
        draws = _run_chain(
            chain_id      = ch + 1,
            cbs           = cbs,
            X             = X,
            hyper         = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                             for k, v in hyper.items()},
            mcmc          = mcmc,
            burnin        = burnin,
            thin          = thin,
            rng           = rng,
            trace         = trace,
            n_mh_steps    = n_mh_steps,
            covariate_cols= ["intercept"] + list(covariates),
        )
        all_lvl1.append(draws["level_1"])
        all_lvl2.append(draws["level_2"])
        all_ll.append(draws["log_likelihood"])

    marginal_ll = float(np.mean(np.concatenate(all_ll, axis=0)))
    return {
        "level_1": all_lvl1,
        "level_2": all_lvl2,
        "log_likelihood": marginal_ll,
    }


def draw_future_transactions(
    cbs: pd.DataFrame,
    draws: Dict[str, Any],
    T_star: float = 39.0,
    *,
    simulate_spend: bool = True,         # ← NEW: set False for counts-only
    sigma_s: float = 0.50,               # ← NEW: log-std of spend per trx
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Posterior predictive simulation for an RFM–M model.

    Parameters
    ----------
    cbs : DataFrame
        Calibration CBS with at least 'T_cal'.
    draws : Dict[str, Any]
        Output from the Gibbs sampler.  Expects level-1 arrays shaped
        (n_draws, n_customers, 5)  with columns (λ, μ, τ, z, η).
    T_star : float
        Forecast horizon length.
    simulate_spend : bool, default True
        If True, also simulate lognormal spend amounts and return
        a tuple (x_future, spend_future).  If False, return x_future only.
    sigma_s : float, default 0.50
        Log-scale standard deviation of spend per transaction.
        Pass your posterior mean or a scalar hyper-parameter.
    seed : int | None
        RNG seed.

    Returns
    -------
    If simulate_spend is False
        np.ndarray  (n_draws, n_customers)  – future transaction counts.
    If simulate_spend is True
        Tuple (x_future, spend_future) – both shaped as above, where
        spend_future is the simulated monetary value per customer.
    """
    rng = np.random.default_rng(seed)

    T_cal = cbs["T_cal"].to_numpy(float)
    lvl1_all_chains = draws["level_1"]

    x_sims, spend_sims = [], []

    for chain in lvl1_all_chains:
        for draw in chain:                                   # (N, 5)
            lambdas, mus, tau, z_flag, eta = draw.T
            alive = z_flag > 0.5

            # (1) Remaining lifetime within the forecast window
            tau_star = np.full_like(T_cal, T_star, dtype=float)
            tau_star[~alive] = np.clip(
                tau[~alive] - T_cal[~alive],
                a_min=0.0,
                a_max=T_star,
            )

            # (2) Future transaction counts
            x_next = rng.poisson(lam=lambdas * tau_star)
            x_sims.append(x_next)

            if simulate_spend:
                # Simulate lognormal spend for each *individual* transaction
                # and aggregate to customer totals.
                #   spend = exp( η_i + ε ),   ε ~ Normal(0, σ_s)
                #
                # To avoid Python loops, build an index vector whose length
                # equals the total number of transactions in this draw.
                #
                if x_next.sum() > 0:
                    idx = np.repeat(np.arange(len(x_next)), x_next)
                    spend_per_trx = rng.lognormal(
                        mean=eta[idx], sigma=sigma_s, size=len(idx)
                    )
                    spend_tot = np.bincount(
                        idx, weights=spend_per_trx, minlength=len(x_next)
                    )
                else:
                    spend_tot = np.zeros_like(x_next, dtype=float)

                spend_sims.append(spend_tot)

    x_future = np.vstack(x_sims)  # (total_draws, N)

    if not simulate_spend:
        return x_future

    spend_future = np.vstack(spend_sims)  # (total_draws, N)
    return x_future, spend_future
