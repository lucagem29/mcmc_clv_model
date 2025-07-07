# %% 1. Import libraries & set up project root
# -- 1. Import libraries & set up project root --
import os
import pandas as pd
import sys
# ------------------------------------------------------------------

# Find project root (folder containing "src") )
cwd = os.getcwd()
while not os.path.isdir(os.path.join(cwd, 'src')):
    parent = os.path.dirname(cwd)
    if parent == cwd:
        break  # Reached the root of the filesystem
    cwd = parent
project_root = cwd
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utility function for converting event log to CBS format
from src.models.utils.elog2cbs2param import elog2cbs
# ------------------------------------------------------------------

# %% 2. Load abe Elog data & convert to CBS format
# -- 2. Load abe Elog data & convert to CBS format --

data_path = os.path.join(project_root, "data", "raw", "cdnow_abeElog.csv")
cdnowElog = pd.read_csv(data_path)

# Convert date column to datetime
cdnowElog["date"] = pd.to_datetime(cdnowElog["date"])

# Convert event log to customer-by-sufficient-statistic (CBS) format
cbs = elog2cbs(cdnowElog, units="W", T_cal="1997-09-30", T_tot="1998-06-30")
#cbs = create_customer_summary(cdnowElog, T_cal="1997-09-30", T_tot="1998-06-30")
cbs = cbs.rename(columns={"t.x": "t_x", "T.cal": "T_cal", "x.star": "x_star"})
if "first" in cbs.columns:
    cbs["first"] = pd.to_datetime(cbs["first"]).dt.date
cbs.head()

# %% 3. Load customer demographics & merge with CBS
# -- 3. Load customer demographics & merge with CBS --

# Load full CBS to enrich with age_scaled and gender_binary
# Run this after 2B_cdnow_elog2cbs_full.py to create full CBS
full_cbs_path = os.path.join(project_root, "data", "processed", "cdnow_fullCBS.csv")
df_full_cbs = pd.read_csv(full_cbs_path, usecols=["cust", "age", "age_scaled", "gender_binary", "first_sales_scaled"], dtype={"cust": str})
# Ensure 'cust' keys share the same dtype
cbs["cust"] = cbs["cust"].astype(str)
# Merge demographic features into CBS
cbs = cbs.merge(df_full_cbs, on="cust", how="left")
# Report how many customers were enriched with demographics
total_cust = len(cbs)
matched_cust = cbs["age_scaled"].notna().sum()
print(f"Enriched demographics for {matched_cust} of {total_cust} customers.")

# Save CBS to processed data folder
processed_dir = os.path.join(project_root, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)
output_path = os.path.join(processed_dir, "cdnow_abeCBS.csv")
cbs.to_csv(output_path, index=False)
print(f"Saved CBS data to {output_path}")
# %%
