"""Run convergence diagnostics for stored MCMC draws.

This script loads the pickled MCMC draws for the bivariate and trivariate
Pareto/NBD models (with and without covariates) and computes basic
convergence metrics:
    * R-hat (Gelman--Rubin statistic)
    * Effective sample size (ESS)
    * Geweke z-scores

It also generates traceplots and autocorrelation plots for the cohort-level
parameters.  Figures are written to ``outputs/figures/convergence`` and a
CSV table with the diagnostics is stored in ``outputs/excel``.  If the pickled
draws are not available (for example because Git LFS files were not fetched),
the script will skip the respective model.
"""

from __future__ import annotations

import os
import pickle
import csv
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from src.utils.project_root import add_project_root_to_sys_path

project_root = add_project_root_to_sys_path()

PICKLE_DIR = os.path.join(project_root, "outputs", "pickles")
FIG_DIR = os.path.join(project_root, "outputs", "figures", "convergence")
CSV_PATH = os.path.join(project_root, "outputs", "excel", "convergence_diagnostics.csv")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)


# ---------------------------------------------------------------------------
# Basic statistics helpers
# ---------------------------------------------------------------------------

def _autocorrelation(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    result = np.correlate(x, x, mode="full")
    ac = result[result.size // 2 :]
    ac /= ac[0]
    return ac


def compute_rhat(draws: np.ndarray) -> np.ndarray:
    """Return R-hat for each parameter.

    Parameters
    ----------
    draws : array, shape (chains, draws, params)
    """
    chains, n, params = draws.shape
    chain_means = draws.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)

    B = n / (chains - 1) * ((chain_means - grand_mean) ** 2).sum(axis=0)
    W = ((draws - chain_means[:, None, :]) ** 2).sum(axis=(0, 1)) / (chains * (n - 1))

    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def compute_ess(draws: np.ndarray) -> np.ndarray:
    """Return the effective sample size for each parameter."""
    chains, n, params = draws.shape
    ess = np.empty(params)

    for p in range(params):
        ac_per_chain = np.array([_autocorrelation(draws[c, :, p]) for c in range(chains)])
        mean_ac = ac_per_chain.mean(axis=0)

        # positive sequence estimator
        t = 1
        s = 0.0
        while t + 1 < mean_ac.size:
            rho = mean_ac[t] + mean_ac[t + 1]
            if rho < 0:
                break
            s += rho
            t += 2
        ess[p] = chains * n / (1 + 2 * s)
    return ess


def geweke_zscore(draws: np.ndarray, first: float = 0.1, last: float = 0.5) -> np.ndarray:
    """Return Geweke z-scores for each chain and parameter."""
    chains, n, params = draws.shape
    a, b = int(first * n), int((1 - last) * n)
    z = np.empty((chains, params))
    for c in range(chains):
        early = draws[c, :a, :]
        late = draws[c, b:, :]
        mean_diff = early.mean(axis=0) - late.mean(axis=0)
        var_early = early.var(axis=0, ddof=1)
        var_late = late.var(axis=0, ddof=1)
        z[c] = mean_diff / np.sqrt(var_early / a + var_late / (n - b))
    return z


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_trace(draws: np.ndarray, param_names: List[str], title_prefix: str) -> None:
    chains, n, params = draws.shape
    for p in range(params):
        plt.figure()
        for c in range(chains):
            plt.plot(draws[c, :, p], label=f"chain {c + 1}", alpha=0.7)
        plt.xlabel("draw")
        plt.ylabel(param_names[p])
        plt.title(f"{title_prefix} – {param_names[p]}")
        plt.legend()
        plt.tight_layout()
        fname = f"{title_prefix}_trace_{p}.png".replace(" ", "_")
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
        plt.close()


def plot_autocorr(draws: np.ndarray, param_names: List[str], title_prefix: str, max_lag: int = 50) -> None:
    chains, n, params = draws.shape
    for p in range(params):
        plt.figure()
        for c in range(chains):
            ac = _autocorrelation(draws[c, :, p])[: max_lag + 1]
            plt.plot(range(max_lag + 1), ac, label=f"chain {c + 1}", marker="o", ls="-")
        plt.xlabel("lag")
        plt.ylabel("autocorrelation")
        plt.title(f"{title_prefix} – {param_names[p]}")
        plt.legend()
        plt.tight_layout()
        fname = f"{title_prefix}_autocorr_{p}.png".replace(" ", "_")
        plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

MODELS = {
    "bivariate_M1": "bivariate_M1.pkl",
    "bivariate_M2": "bivariate_M2.pkl",
    "trivariate_M1": "trivariate_M1.pkl",
    "trivariate_M2": "trivariate_M2.pkl",
}


def load_draws(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    lvl2 = np.array(obj["level_2"])
    return lvl2


def main() -> None:
    table_rows: List[List[float | str]] = []
    headers: List[str] = []
    wrote_header = False

    for model, file_name in MODELS.items():
        file_path = os.path.join(PICKLE_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found – skipping {model}")
            continue

        draws = load_draws(file_path)
        param_names = [f"param_{i}" for i in range(draws.shape[-1])]

        rhat = compute_rhat(draws)
        ess = compute_ess(draws)
        geweke = geweke_zscore(draws)

        if not wrote_header:
            headers = ["model", "parameter", "rhat", "ess"] + [
                f"geweke_chain{c+1}" for c in range(geweke.shape[0])
            ]
            wrote_header = True

        print(f"\n=== {model} ===")
        for i, name in enumerate(param_names):
            print(f"{name}: R-hat={rhat[i]:.3f} | ESS={ess[i]:.1f}")
            row = [model, name, float(rhat[i]), float(ess[i])] + [
                float(geweke[c, i]) for c in range(geweke.shape[0])
            ]
            table_rows.append(row)
        print("Geweke z-score (first chain):", np.round(geweke[0], 2))

        plot_trace(draws, param_names, model)
        plot_autocorr(draws, param_names, model)

    if table_rows:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(table_rows)


if __name__ == "__main__":
    main()
