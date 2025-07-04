# Abe (2009) Pareto/NBD Model Implementation (Modular Python Version)

This repository contains a modular Python implementation of the hierarchical Bayesian Pareto/NBD model from Abe (2009), applied to the CDNOW dataset. It supports model estimation, customer-level forecasts, and evaluation of model fit and predictive accuracy, reproducing Tables 1–4 and key figures from the original paper.
Furthermore, it contains the three parameter extention, suggested by Abe 2015.

## Repository Structure

```
Abe_MCMC/
├── data/
│   ├── raw/                  # Raw input data
│   └── processed/            # Processed data
├── outputs/
│   ├── excel/                # Generated Excel summaries and tables
│   ├── figures/              # Generated figures and plots of abe data --> TO DO TO UPDATE IN SUBFOLDER
│   │   ├── full_cdnow/       # containing plots of individual analyis on full dataset
│   │   └── full_cdnow_both/  # containing plots of four model comparison
│   └── pickles/              # Saved MCMC draws and model outputs
├── src/
│   ├── data_processing/
│   │   ├── cdnow_abe_covariates.py  # Adding covariates to Abes dataset (1/10 of full)
│   │   └── cdnow_full.py     # Processing the full CDNOW dataset
│   ├── models/
│   │   ├── bivariate/
│   │   │   ├── mcmc.py       # Bivariate MCMC routines
│   │   │   ├── analysis.py   # Bivariate model analysis and plotting
│   │   │   └── run_mcmc.py   # Script to run MCMC and save results
│   │   ├── trivariate/
│   │   │   ├── mcmc.py       # Trivariate MCMC routines
│   │   │   ├── analysis.py   # Trivariate model analysis and plotting
│   │   │   └── run_mcmc.py   # Script to run MCMC and save results
│   │   └── utils/
│   │       └── elog2cbs2param.py # Utility: event log to CBS conversion
│   └── full_analysis.py      # Compiling the graphs of the four models on full CDNOW dataset
├── compare_pickles.py        # sanity check to make sure pickles do differ
├── detailed_analysis_pickles # second sanity check to really make sure
├── README.md                 
└── SETUP_REQUIREMENTS.md     # Setup requirements !!
```

## Prerequisites

- Python 3.8 or higher
- Required packages:
  ```bash
  pip install numpy pandas scipy matplotlib seaborn jupyter openpyxl lifetimes arviz
  ```

## Usage

### 1. Data Preparation

- Place the raw CDNOW event log (e.g., `cdnowElog.csv`) in `data/raw/`.
- Use the data processing script to convert the event log to CBS format:
  ```bash
  python src/data_processing/cdnow_abe_covariates.py
  ```
  This will output a processed file (e.g., `cdnow_cbs_customers.csv`) in `data/processed/`.

### 2. Model Estimation and Analysis

#### Bivariate Model
- Run MCMC and analysis (from project root):
  ```bash
  python -m src.models.bivariate.analysis
  ```
  This will:
  - Estimate bivariate models (with/without covariates)
  - Save MCMC draws to `outputs/pickles/`
  - Save tables to `excel/`
  - Save figures to `outputs/figures/`

#### Trivariate Model
- Run MCMC and analysis (from project root):
  ```bash
  python -m src.models.trivariate.analysis
  ```
  (Outputs are saved in the same way as above, but for trivariate models.)

### 3. Interactive Exploration
- Launch a Jupyter notebook for custom analysis:
  ```bash
  jupyter notebook
  ```
  and open any notebook in the `notebooks/` directory (if present).

### Convergence Diagnostics

If you have the pickled MCMC draws available, you can compute basic
convergence statistics and generate trace/autocorrelation plots:

```bash
python -m src.convergence_diagnostics
```

The resulting figures will be written to `outputs/figures/convergence/`.

## Results

- **Excel summaries**: All tables (Tables 1–4) are saved in `excel/`.
- **Pickled MCMC draws**: Saved in `outputs/pickles/` for reproducibility and further analysis.
- **Figures**: All plots and figures are saved in `outputs/figures/`.

## Figures

- **Figure 2**: Weekly Time-Series Tracking for CDNOW Data
- **Figure 3**: Conditional Expectation of Future Transactions
- **Figure 4**: Scatter Plot of Posterior Means of λ and μ
- **Figure 5**: Distribution of log(λ)–log(μ) Correlations
- **Scatter M1 vs. M2**: Actual vs. Predicted x_star (M1 vs. M2)
- **Alive vs. Churned**: Predicted Alive vs. Churned Customers

(Figures are saved in `outputs/figures/`)

## Contact

For questions or contributions, please open an issue or submit a pull request.  