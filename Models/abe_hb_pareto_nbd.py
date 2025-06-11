# abe_hb_pareto_nbd.py
"""Hierarchical Bayes extension of the Pareto/NBD model (Abe 2009).

Implements the MCMC algorithm described in Section 3.3 of Abe (2009) and the
Technical Appendix using only NumPy & SciPy.  The code is deliberately
modular so that individual components (samplers, likelihood, diagnostics) can
be unit-tested and swapped out for faster back-ends such as Numba or JAX.

The public interface is two classes:

* `CustomerRFT` - lightweight container holding recency/frequency data for a
  single customer (x, t_x, T) and optional covariates d_i.
* `AbeHBParetoNBD` - run the full Gibbs / Metropolis-within-Gibbs sampler,
  expose posterior draws and convenience methods for CLV-style statistics.

Example
-------
>>> import pandas as pd
>>> from abe_hb_pareto_nbd import AbeHBParetoNBD, load_cdnow
>>> rf, cov = load_cdnow()
>>> model = AbeHBParetoNBD(rf, cov)
>>> model.run_mcmc(14_000, burn_in=10_000, thin=5)
>>> ax = model.plot_cumulative_transactions();
>>> model.summary_table().to_csv("fit_metrics.csv", index=False)

References
----------
* Abe, M. (2009).  "Counting Your Customers One by One: A Hierarchical Bayes
  Extension to the Pareto/NBD Model", *Marketing Science*, 28(3), 541-553.
* Abe, M. (2009).  Online Technical Appendix.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import numpy.random as npr
import pandas as pd
from scipy import stats

# -------------------------------------------------------------------------
# Optional speed‑up via Numba JIT -----------------------------------------
# -------------------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:                # Numba not installed → graceful fallback
    _HAVE_NUMBA = False

# Inform the user which backend is active
if _HAVE_NUMBA:
    print("[AbeHB] Numba JIT backend active.")
else:
    print("[AbeHB] Numba not found - using pure NumPy/SciPy.")

# -------------------------------------------------------------------------
# Optional pretty progress bar --------------------------------------------
# -------------------------------------------------------------------------
try:
    from tqdm import trange          # progress bar for loops
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

# Fallback: use built‑in range when tqdm is unavailable
if not _HAVE_TQDM:
    def trange(*args, **kwargs):
        return range(*args)

if _HAVE_NUMBA:
    @njit
    def _p_alive_numba(lam, mu, T, t_x):
        """P[z_i = 1 | λ, μ] - Eq. 6 - compiled with Numba."""
        return 1.0 / (1.0 + (mu / (lam + mu)) * (np.exp((lam + mu) * (T - t_x)) - 1.0))

    @njit
    def _trunc_exp_numba(rate, lower, upper, u):
        """Draw from Exp(rate) truncated to (lower, upper) given u~U(0,1)."""
        return -np.log(np.exp(-rate * upper) + u * (np.exp(-rate * lower) - np.exp(-rate * upper))) / rate

    # full JIT log‑likelihood and parallel customer update (new in numba_backend.py)
    try:
        from Models.numba_backend import (
            log_complete_lik_numba,
            update_customers,        # full parallel customer sweep
        )
    except ImportError:
        log_complete_lik_numba = None
        update_customers = None

# ----------------------------------------------------------------------------
# Helper dataclass -----------------------------------------------------------
# ----------------------------------------------------------------------------

@dataclass
class CustomerRFT:
    """Container for a single customer's recency/frequency data."""

    x: int           # number of repeat purchases in (0, T]
    t_x: float       # time of last purchase (recency) 0 ≤ t_x ≤ T
    T: float         # length of calibration window
    d: np.ndarray    # K‑dim covariate vector (include intercept!)

# ----------------------------------------------------------------------------
# Main model class -----------------------------------------------------------
# ----------------------------------------------------------------------------

class AbeHBParetoNBD:
    """Hierarchical-Bayes Pareto/NBD (Abe 2009) implemented from scratch."""

    def __init__(self, customers: List[CustomerRFT],
                 K: int | None = None,
                 beta0: np.ndarray | None = None,
                 Sigma0: np.ndarray | None = None,
                 nu0: float = 4.0,
                 Gamma0: np.ndarray | None = None,
                 seed: int | None = None):
        r"""Parameters
        ----------
        customers
            List of ``CustomerRFT`` instances.
        K
            Dimension of covariate vector (defaults to d.shape[0] of first
            customer).
        beta0, Sigma0
            Mean and covariance of the MVN prior on population regression
            coefficients β.  If ``None`` → diffuse priors.
        nu0, Gamma0
            Degrees of freedom and scale matrix of the inverse-Wishart prior on
            Γ₀ (population residual covariance).
        seed
            RNG seed for reproducibility.
        """
        self.customers = customers
        self.N = len(customers)
        self.K = K or customers[0].d.size
        self.rng = npr.default_rng(seed)

        # set diffuse hyper‑priors if not provided
        self.beta0 = np.zeros((self.K, 2)) if beta0 is None else beta0
        # prior covariance for the K covariates (I₂ ⊗ Σ₀ will be formed later)
        self.Sigma0 = np.eye(self.K) * 1e6 if Sigma0 is None else Sigma0
        self.nu0 = nu0
        self.Gamma0 = np.eye(2) * 10.0 if Gamma0 is None else Gamma0

        # MCMC state (initialised later)
        self.theta = np.zeros((self.N, 2))  # log‑λ, log‑μ per customer
        self.z = np.ones(self.N, dtype=int)  # active indicator
        self.y = np.full(self.N, np.nan)     # dropout time for inactive
        self.beta = np.zeros((self.K, 2))
        self.Gamma = np.eye(2)
        self.draws: Dict[str, List[np.ndarray]] = {}

        # cache static per‑customer arrays for the parallel Numba kernel
        self._x_arr   = np.array([c.x   for c in customers], dtype=np.int64)
        self._t_x_arr = np.array([c.t_x for c in customers], dtype=np.float64)
        self._T_arr   = np.array([c.T   for c in customers], dtype=np.float64)
        self._D_arr   = np.vstack([c.d  for c in customers]).astype(np.float64)

    # ---------------------------------------------------------------------
    # Public API -----------------------------------------------------------
    # ---------------------------------------------------------------------

    def run_mcmc(self, n_iter: int = 20_000, *, burn_in: int = 5_000,
                 thin: int = 10, proposal_scale: float = 0.3) -> None:
        print(f"[AbeHB] Running chain with {'Numba' if _HAVE_NUMBA else 'pure NumPy'} backend.")
        """Run the Gibbs / Metropolis-within-Gibbs sampler."""
        print(f"[AbeHB] Starting MCMC: {n_iter:,} iterations "
              f"(burn‑in {burn_in:,}, thin {thin}) on {self.N:,} customers.")
        # initial values – log‑normal centred on simple heuristics
        for i, cust in enumerate(self.customers):
            lam0 = (cust.x + 1) / (cust.T + 1e-3)
            mu0 = 0.1
            self.theta[i] = np.log([lam0, mu0])
        self.beta = np.zeros((self.K, 2))
        self.Gamma = np.eye(2)

        accept = 0
        iterator = trange(1, n_iter + 1, desc="MCMC", disable=not _HAVE_TQDM)
        for it in iterator:
            # Step 2: update customer‑specific states --------------------
            if _HAVE_NUMBA and update_customers is not None:
                # recompute Γ⁻¹ and log|Γ| with current Γ
                Ginv   = np.linalg.inv(self.Gamma)
                logdetG = np.log(np.linalg.det(self.Gamma))

                rng_norm = self.rng.normal(size=2*self.N).astype(np.float64)
                rng_uni  = self.rng.random(size=3*self.N).astype(np.float64)

                accept += update_customers(
                    self.theta, self.z, self.y, self.beta,
                    Ginv, logdetG,
                    self._x_arr, self._t_x_arr, self._T_arr, self._D_arr,
                    proposal_scale,
                    rng_norm, rng_uni,
                )
            else:
                for i, cust in enumerate(self.customers):
                    lam_i, mu_i = np.exp(self.theta[i])
                    # 2a sample z_i (Equation 6)
                    p_alive = self._p_alive(lam_i, mu_i, cust)
                    self.z[i] = self.rng.binomial(1, p_alive)

                    # 2b if inactive sample y_i
                    if self.z[i] == 0:
                        self.y[i] = cust.t_x + self._trunc_exp(lam_i + mu_i,
                                                              lower=0.0,
                                                              upper=cust.T - cust.t_x)
                    else:
                        self.y[i] = np.nan

                    # 2c RW‑Metropolis on θ_i
                    theta_prop = self.theta[i] + self.rng.normal(0, proposal_scale, 2)
                    log_acc_ratio = (self._log_complete_lik(theta_prop, cust, i) -
                                     self._log_complete_lik(self.theta[i], cust, i))
                    if np.log(self.rng.random()) < log_acc_ratio:
                        self.theta[i] = theta_prop
                        accept += 1

            # Step 3: multivariate regression update for (β, Γ)
            D = np.vstack([c.d for c in self.customers])       # N×K
            resid = self.theta - D @ self.beta                 # N×2
            # sample Γ from inverse‑Wishart
            S = resid.T @ resid
            self.Gamma = stats.invwishart.rvs(df=self.nu0 + self.N,
                                             scale=self.Gamma0 + S,
                                             random_state=self.rng)
            # sample β conditional on Γ
            V_beta = np.linalg.inv(np.kron(self.Gamma, np.eye(self.K)) +
                                   np.kron(np.eye(2), self.Sigma0))
            m_beta = V_beta @ (
                np.kron(np.eye(2), self.Sigma0) @ self.beta0.ravel()
                + np.kron(self.Gamma, D.T) @ self.theta.ravel()
            )
            beta_vec = self.rng.multivariate_normal(m_beta, V_beta)
            self.beta = beta_vec.reshape(self.K, 2)

            # Step 4: draw new θ from population for PPP diagnostics (optional)
            # skipped in this minimal implementation

            # store draws after burn‑in/thinning
            if it > burn_in and (it - burn_in) % thin == 0:
                self._store_draw()

            #if it % 1000 == 0:
            #    acc_rate = accept / (self.N * 1000)
            #    accept = 0
            #    print(f"iter {it:>6}  acceptance ≈ {acc_rate:.3f}")

        print(f"[AbeHB] MCMC finished – stored {len(self.draws['theta']):,} posterior draws.")

    # ------------------------------------------------------------------
    # Convenience methods ----------------------------------------------
    # ------------------------------------------------------------------

    def posterior_means(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return posterior means of λ and μ (length N).

        Each retained draw of θ is stored as an (N×2) array inside
        `self.draws["theta"]`, so `self.draws["theta"]` is a list whose length
        equals the number of posterior samples.  We first stack that list into
        a 3‑D array to compute vectorised averages.
        """
        if not self.draws.get("theta"):
            raise RuntimeError("No posterior draws stored – run run_mcmc() first.")

        theta_arr = np.stack(self.draws["theta"], axis=0)   # shape (n_draws, N, 2)
        lam = np.exp(theta_arr[:, :, 0]).mean(axis=0)       # posterior mean λ_i
        mu  = np.exp(theta_arr[:, :, 1]).mean(axis=0)       # posterior mean μ_i
        return lam, mu

    def expected_transactions(self, weeks: float = 39.0) -> np.ndarray:
        """Compute E[X(t)] for each customer at *future* horizon *weeks* (Equation 8)."""
        lam, mu = self.posterior_means()
        return lam / (mu) * (1 - np.exp(-mu * weeks))

    # plotting helpers are implemented in a separate file `plots.py`

    # ------------------------------------------------------------------
    # Internal utils ----------------------------------------------------
    # ------------------------------------------------------------------

    def _store_draw(self) -> None:
        for name, arr in (("theta", self.theta.copy()),
                          ("beta", self.beta.copy()),
                          ("Gamma", self.Gamma.copy())):
            self.draws.setdefault(name, []).append(arr)

    # ---- probability that customer is *alive* at T (Eq. 6) ------------
    @staticmethod
    def _p_alive(lam: float, mu: float, cust: CustomerRFT) -> float:
        if _HAVE_NUMBA:                       # use JIT version if available
            return _p_alive_numba(lam, mu, cust.T, cust.t_x)
        denom = 1.0 + (mu / (lam + mu)) * (np.exp((lam + mu) * (cust.T - cust.t_x)) - 1.0)
        return 1.0 / denom

    # ---- truncated exponential sampler --------------------------------
    def _trunc_exp(self, rate: float, *, lower: float, upper: float) -> float:
        if _HAVE_NUMBA:                       # faster, JIT‑compiled path
            u = self.rng.random()
            return _trunc_exp_numba(rate, lower, upper, u)
        u = self.rng.uniform(np.exp(-rate * upper), np.exp(-rate * lower))
        return -np.log(u) / rate

    # ---- complete log‑likelihood for customer i -----------------------
    def _log_complete_lik(self, theta: np.ndarray, cust: CustomerRFT, idx: int) -> float:
        lam, mu = np.exp(theta)
        x, tx, T = cust.x, cust.t_x, cust.T
        z = self.z[idx]
        y = self.y[idx]
        # likelihood part (Eq. 5)
        if x > 0:
            ll = (x * np.log(lam) + (x - 1) * np.log(tx) - math.lgamma(x) - lam * tx)
        else:
            ll = 0.0  # no repeat purchase term
        if z == 1:
            ll += -(lam + mu) * (T - tx)
        else:
            ll += np.log(mu) - (lam + mu) * (y - tx) - (lam + mu) * (T - y)
        # prior part MVN(log‑λ, log‑μ | β′d, Γ)
        diff = theta - (cust.d @ self.beta)
        ll += stats.multivariate_normal.logpdf(diff, mean=np.zeros(2), cov=self.Gamma)
        return ll

# ----------------------------------------------------------------------------
# Utility functions -----------------------------------------------------------
# ----------------------------------------------------------------------------

_CDNOW_URL = "https://raw.githubusercontent.com/CamDavidsonPilon/lifetimes/master/lifetimes/datasets/CDNOW_master.txt"

_def_cov_cols = ["intercept", "init_amount"]


def load_cdnow(path: str | pathlib.Path | None = None) -> Tuple[List[CustomerRFT], pd.DataFrame]:
    """Load the CDNOW dataset and convert to R/F/T representation.

    Parameters
    ----------
    path
        Path to *CDNOW_master.txt* (if ``None`` download from GitHub).
    """
    if path is None:
        path = pathlib.Path.home() / ".cache" / "cdnow_master.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            import urllib.request
            urllib.request.urlretrieve(_CDNOW_URL, path)

    # CDNOW_master.txt is whitespace‑delimited with four columns: id, yyyymmdd, qty, spend
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=1,                 # ← NEW: skip "customer_id date number_of_cds ..." header
        names=["id", "date", "qty", "spend"],
        dtype={"id": str, "date": str, "qty": int, "spend": float},
    )
    # convert the yyyymmdd string to a proper Timestamp
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d")

    # recency/frequency transformation (Abe uses weeks)
    first_txn = raw.groupby("id")['date'].min()
    raw = raw.join(first_txn, on="id", rsuffix="_first")
    raw["t"] = (raw["date"] - raw["date_first"]).dt.days / 7.0

    rft_rows = []
    cov_rows = []
    for cid, grp in raw.groupby("id"):
        T_cal = 39.0  # weeks as in Abe (2009)
        in_cal = grp[grp["t"] <= T_cal]
        if in_cal.empty:
            continue
        x = len(in_cal) - 1
        t_x = in_cal["t"].max() if x > 0 else 0.0
        init_amt = grp.loc[grp["t"] == 0, "spend"].iloc[0]
        d = np.array([1.0, init_amt / 1000.0])  # intercept + scaled cov
        rft_rows.append(CustomerRFT(x, t_x, T_cal, d))
        cov_rows.append(dict(id=cid, intercept=1.0, init_amount=init_amt))

    return rft_rows, pd.DataFrame(cov_rows)
