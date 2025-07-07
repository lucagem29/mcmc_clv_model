# %% 1. Import libraries & set up project root
# -- 1. Import libraries & set up project root --
import os
import sys
import pandas as pd

# ------------------------------------------------------------------
# Find project root (folder containing "src") )
# ------------------------------------------------------------------
cwd = os.getcwd()
while not os.path.isdir(os.path.join(cwd, 'src')):
    parent = os.path.dirname(cwd)
    if parent == cwd:
        break  # Reached the root of the filesystem
    cwd = parent
project_root = cwd
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utility function for converting event log to CBS forma
from src.models.utils.elog2cbs2param import elog2cbs

# Make sure output directory exists
output_dir = os.path.join(project_root, "data", "raw")
os.makedirs(output_dir, exist_ok=True)
# ------------------------------------------------------------------

# %% 2. Load full Elog data & convert to CBS format
# -- 2. Load full Elog data & convert to CBS format --

data_path = os.path.join(project_root, "data", "raw", "cdnow_fullElog.csv")
cdnow_full_Elog = pd.read_csv(data_path)

# Convert date column to datetime
cdnow_full_Elog["date"] = pd.to_datetime(cdnow_full_Elog["date"])

# Convert event log to customer-by-sufficient-statistic (CBS) format
cbs_ful = elog2cbs(cdnow_full_Elog, units="W", T_cal="1997-09-30", T_tot="1998-06-30")
#cbs = create_customer_summary(cdnowElog, T_cal="1997-09-30", T_tot="1998-06-30")
cbs_ful = cbs_ful.rename(columns={"t.x": "t_x", "T.cal": "T_cal", "x.star": "x_star"})
if "first" in cbs_ful.columns:
    cbs_ful["first"] = pd.to_datetime(cbs_ful["first"]).dt.date
cbs_ful.head()

# Load customer demographics & merge with CBS
data_path = os.path.join(project_root, "data", "raw", "cdnow_fullCovar.csv")
df_customers = pd.read_csv(data_path)

df_cbs__full_customers = cbs_ful.merge(
    df_customers,
    left_on="cust",
    right_on="cust",
    how="left"
)

print("CBS merged with customer demographics:")
print(df_cbs__full_customers.head())

# Some data processing


# Compute each customer's first purchase amount and scale
first = (
    cdnow_full_Elog
    .groupby("cust")["sales"]
    .first()
    .reset_index()
    .rename(columns={"sales": "first_sales"})
)
# Scale to $10^-3
first["first_sales"] = first["first_sales"] * 1e-3

# Merge first_sales into the full CBS DataFrame
df_cbs__full_customers = df_cbs__full_customers.merge(
    first[["cust", "first_sales"]],
    on="cust",
    how="left"
)

# Normalize first_sales
mean_fs = df_cbs__full_customers["first_sales"].mean()
std_fs  = df_cbs__full_customers["first_sales"].std()
df_cbs__full_customers["first_sales_scaled"] = (
    df_cbs__full_customers["first_sales"] - mean_fs
) / std_fs


# Scale age (continuous)
df_cbs__full_customers['age_scaled'] = (
    df_cbs__full_customers['age'] - df_cbs__full_customers['age'].mean()
) / df_cbs__full_customers['age'].std()

#One-hot encode zone and state
#df_cbs_customers = pd.get_dummies(
#    df_cbs_customers,
#    columns=['zone', 'state'],
#    prefix=['zone', 'state']
#)'

# Binary encode gender (map to 0/1)
df_cbs__full_customers['gender_binary'] = df_cbs__full_customers['gender'].map({
    'M': 1,
    'F': 0
})
df_cbs__full_customers.drop(columns=['gender', 'zone', 'state', 'age_category', 'first_sales'], inplace=True)

print(df_cbs__full_customers.head())
# ------------------------------------------------------------------

# %% 3. Save the enriched full CBS
# -- 3. Save the enriched full CBS --
output_dir = os.path.join(project_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)
cbs_merge_path = os.path.join(output_dir, "cdnow_fullCBS.csv")
df_cbs__full_customers.to_csv(cbs_merge_path, index=False)
print(f"Saved CBS with customer demographics to {cbs_merge_path}")
# ------------------------------------------------------------------

# %%
