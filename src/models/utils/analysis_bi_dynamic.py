"""
Utility for generating parameter names and labels for bivariate hierarchical models with arbitrary covariates.
"""
from typing import List, Tuple

def build_bivariate_param_names_and_labels(covariate_cols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Generate internal parameter names and human-readable labels
    for a bivariate hierarchical model with the given covariates.

    Args:
        covariate_cols (List[str]): List of covariate names (e.g., ['age', 'income']).

    Returns:
        Tuple[List[str], List[str]]: Tuple of (internal param names, human-readable labels)
    """
    # Internal parameter names
    lambda_names = ["log_lambda (intercept)"] + [f"log_lambda ({cov})" for cov in covariate_cols]
    mu_names     = ["log_mu (intercept)"]     + [f"log_mu ({cov})"     for cov in covariate_cols]
    misc_names   = ["var_log_lambda", "var_log_mu", "cov_log_lambda_mu"]
    param_names  = lambda_names + mu_names + misc_names

    # Human-readable labels for summary index
    lambda_labels = ["Purchase rate log(λ) - Intercept"] + [
        f"Purchase rate log(λ) - {cov.replace('_', ' ')}" for cov in covariate_cols
    ]
    mu_labels     = ["Dropout rate log(μ) - Intercept"] + [
        f"Dropout rate log(μ) - {cov.replace('_', ' ')}" for cov in covariate_cols
    ]
    misc_labels   = [
        "sigma^2_λ = var[log λ]",
        "sigma^2_μ = var[log μ]",
        "sigma_λ_μ = cov[log λ, log μ]"
    ]
    labels = lambda_labels + mu_labels + misc_labels

    return param_names, labels

if __name__ == "__main__":
    # Example usage
    covariates = ["age", "income", "gender"]
    param_names, labels = build_bivariate_param_names_and_labels(covariates)
    print("Parameter names:")
    for name in param_names:
        print("  ", name)
    print("\nLabels:")
    for label in labels:
        print("  ", label)