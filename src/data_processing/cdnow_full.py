# ------------------------------------------------------------------
# this script loads and processes the whole CDNOW dataset
# ------------------------------------------------------------------
#%%
import requests, io
import sys
import os
from src.utils.project_root import add_project_root_to_sys_path
project_root = add_project_root_to_sys_path()
import pandas as pd
# ------------------------------------------------------------------

from src.models.utils.elog2cbs2param import elog2cbs

# Make sure output directory exists
output_dir = os.path.join(project_root, "data", "raw")
os.makedirs(output_dir, exist_ok=True)
# ------------------------------------------------------------------
#%% 1. Download and save data from Hugging Face

base = "https://huggingface.co/datasets/ZennyKenny/CDNOW/resolve/main/"
files = ["purchases.csv", "customers.csv"]

dfs = {}
for fn in files:
    r = requests.get(base + fn)
    r.raise_for_status()
    dfs[fn] = pd.read_csv(io.StringIO(r.text), parse_dates=["date"] if fn=="purchases.csv" else None)

df_purchases = dfs["purchases.csv"]
df_customers = dfs["customers.csv"]
# Rename users_id to cust for consistency with CBS data
df_customers.rename(columns={"users_id": "cust"}, inplace=True)
df_purchases.rename(columns={"users_id": "cust", "amt": "sales"}, inplace=True)

df_customers.describe()
df_purchases.describe()

# Save customer characteristics df locally
df_customers.to_csv(os.path.join(output_dir, "cdnow_customers.csv"), index=False)
df_purchases.to_csv(os.path.join(output_dir, "cdnow_purchases.csv"), index=False)

# %% 2. Load dataset and convert to CBS format
# ------ 2. Load dataset and convert to CBS format ------
data_path = os.path.join(project_root, "data", "raw", "cdnow_purchases.csv")
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

#%% 3. Merge CBS with customer demographics

df_cbs__full_customers = cbs_ful.merge(
    df_customers,
    left_on="cust",
    right_on="id",
    how="left"
).drop(columns=["id"])

print("CBS merged with customer demographics:")
print(df_cbs__full_customers.head())

#%% 4. Some data processing

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
df_cbs__full_customers.drop(columns=['gender', 'zone', 'state', 'age', 'age_category'], inplace=True)

print(df_cbs__full_customers.head())

#%% 5. Save the enriched CBS
output_dir = os.path.join(project_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)
cbs_merge_path = os.path.join(output_dir, "cdnow_cbs_full.csv")
df_cbs__full_customers.to_csv(cbs_merge_path, index=False)
print(f"Saved CBS with customer demographics to {cbs_merge_path}")

# %%