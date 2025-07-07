# Abe (2009) Pareto/NBD Model Implementation (Modular Python Version)

This repository contains a modular Python implementation of the hierarchical Bayesian Pareto/NBD model from Abe (2009), applied to the CDNOW dataset. It supports model estimation, customer-level forecasts, and evaluation of model fit and predictive accuracy, reproducing Tables 1–4 and key figures from the original paper.
Furthermore, it contains the three parameter extension, suggested by Abe 2015.

## Repository Structure

```
Abe_MCMC/
├── data/
│   ├── raw/                  # Raw input data
│   └── processed/            # Processed data
├── outputs/
│   ├── excel/                # Generated Excel summaries and tables
│   ├── figures/              # Generated figures and plots
│   │   ├── abe_replication/  # Plots for Abe subset analysis
│   │   ├── full_extention/   # Plots for full dataset analysis
│   │   ├── full_cdnow/       # Plots for individual analysis on full dataset
│   │   └── full_cdnow_both/  # Plots for four model comparison
│   └── pickles/              # Saved MCMC draws and model outputs
├── src/
│   ├── data_processing/
│   │   ├── cdnow_abe_covariates.py  # Adding covariates to Abe's dataset (1/10 of full)
│   │   ├── cdnow_abe.py      # Extraction of original Abe dataset
│   │   └── cdnow_full.py     # Processing the full CDNOW dataset
│   ├── models/
│   │   ├── bivariate/
│   │   │   ├── mcmc.py       # Bivariate MCMC routines
│   │   │   ├── analysis_full.py   # Bivariate model analysis and plotting (full dataset)
│   │   │   ├── analysis_abe.py    # Bivariate model analysis and plotting (Abe subset)
│   │   │   └── run_mcmc.py   # Script to run MCMC and save results
│   │   ├── trivariate/
│   │   │   ├── mcmc.py       # Trivariate MCMC routines
│   │   │   ├── analysis.py   # Trivariate model analysis and plotting
│   │   │   └── run_mcmc.py   # Script to run MCMC and save results
│   │   └── utils/
│   │       ├── elog2cbs2param.py # Utility: event log to CBS conversion
│   │       └── analysis_bi_dynamic.py # Utility: dynamic parameter/label builder for bivariate models
│   └── full_analysis.py      # Compiling the graphs of the four models on full CDNOW dataset
├── full_analyis.py           # Comparing the four models on full dataset
├── pickles_compare.py        # Sanity check to make sure pickles do differ
├── pickles_detailed_analysis.py # Second sanity check to really make sure
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
  python src/data_processing/cdnow_plus.py
  ```
  This will output a processed file (e.g., `cdnow_cbs_customers.csv`) in `data/processed/`.

#### Choosing the Analysis Subset
- For the **Abe subset** (1/10th of the full data, as in Abe 2009), use:
  - `data/processed/cdnow_abeCBS.csv` (CBS)
  - `data/raw/cdnow_abeElog.csv` (event log)
- For the **full dataset** analysis, use:
  - `data/processed/cdnow_fullCBS.csv` (CBS)
  - `data/raw/cdnow_fullElog.csv` (event log)
- Make sure the correct files are loaded in your analysis scripts (see debug prints in scripts for confirmation).

### 2. Model Estimation and Analysis

#### Bivariate Model
- Run MCMC and analysis (from project root):
  ```bash
  python -m src.models.bivariate.analysis_full   # for full dataset
  python -m src.models.bivariate.analysis_abe    # for Abe subset
  ```
  This will:
  - Estimate bivariate models (with/without covariates)
  - Save MCMC draws to `outputs/pickles/`
  - Save tables to `outputs/excel/`
  - Save figures to `outputs/figures/`

#### Trivariate Model
- Run MCMC and analysis (from project root):
  ```bash
  python -m src.models.trivariate.analysis
  ```
  (Outputs are saved in the same way as above, but for trivariate models.)

### 3. Dynamic Covariate Handling

- The code now uses a dynamic parameter/label builder (`build_bivariate_param_names_and_labels` in `src/models/utils/analysis_bi_dynamic.py`).
- You only need to specify your covariate list in one place; all parameter names and summary labels will update automatically throughout the analysis and plotting scripts.
- This makes it easy to extend to models with any number of covariates.

### 4. Interactive Exploration
- Launch a Jupyter notebook for custom analysis:
  ```bash
  jupyter notebook
  ```
  and open any notebook in the `notebooks/` directory (if present).

## Results

- **Excel summaries**: All tables (Tables 1–4) are saved in `outputs/excel/`.
- **Pickled MCMC draws**: Saved in `outputs/pickles/` for reproducibility and further analysis.
- **Figures**: All plots and figures are saved in `outputs/figures/`.

## Figures

- **Figure 2**: Weekly Time-Series Tracking for CDNOW Data
- **Figure 3**: Conditional Expectation of Future Transactions
- **Figure 4**: Scatter Plot of Posterior Means of λ and μ
- **Figure 5**: Distribution of log(λ)–log(μ) Correlations
- **Scatter M1 vs. M2**: Actual vs. Predicted x_star (M1 vs. M2)
- **Alive vs. Churned**: Predicted Alive vs. Churned Customers

(Figures are saved in `outputs/figures/` and subfolders.)

## Troubleshooting & Debugging

- **Figure issues:** If a figure does not display as expected, check the debug print statements in the analysis scripts. These will show the first few values and lengths of the arrays being plotted, as well as the file paths of the data being loaded.
- **Parameter mismatch errors:** If you get a ValueError about shape mismatch in summary tables, ensure your covariate list matches the model used to generate the MCMC draws.
- **Data file selection:** Always confirm the correct CBS and event log files are loaded for your intended analysis (see debug prints).

## Contact

For questions or contributions, please open an issue or submit a pull request.  