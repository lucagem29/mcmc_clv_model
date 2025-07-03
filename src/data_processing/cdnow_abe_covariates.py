# ------------------------------------------------------------------
# this script loads and processes the whole CDNOW dataset
# ------------------------------------------------------------------
#%%
import requests, io
import os
import sys
import pandas as pd


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

from src.models.utils.elog2cbs2param import elog2cbs

# Make sure output directory exists
output_dir = os.path.join(project_root, "data", "raw")
os.makedirs(output_dir, exist_ok=True)

#%% 1. Download and save data from Hugging Face

base = "https://huggingface.co/datasets/ZennyKenny/CDNOW/resolve/main/"
files = ["purchases.csv", "customers.csv"]

dfs = {}
for fn in files:
    r = requests.get(base + fn)
    r.raise_for_status()
    dfs[fn] = pd.read_csv(io.StringIO(r.text), parse_dates=["date"] if fn=="purchases.csv" else None)

df_purchases = dfs["purchases.csv"] # Redundant, as it is 1:1 of our initial dataset
df_customers = dfs["customers.csv"]

df_customers.describe()
df_purchases.describe()

# Save customer characteristics df locally
df_customers.to_csv(os.path.join(output_dir, "cdnow_customers.csv"), index=False)

# %% 2. Load dataset and convert to CBS format
# ------ 2. Load dataset and convert to CBS format ------
# We use dataset available in the BTYD package in R
data_path = os.path.join(project_root, "data", "processed", "cdnowElog.csv")
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

#%% 3. Merge CBS with customer demographics

df_cbs_customers = cbs.merge(
    df_customers,
    left_on="cust",
    right_on="id",
    how="left"
).drop(columns=["id"])

print("CBS merged with customer demographics:")
print(df_cbs_customers.head())

#%% 4. Some data processing

# Scale age (continuous)
df_cbs_customers['age_scaled'] = (
    df_cbs_customers['age'] - df_cbs_customers['age'].mean()
) / df_cbs_customers['age'].std()

#One-hot encode zone and state
#df_cbs_customers = pd.get_dummies(
#    df_cbs_customers,
#    columns=['zone', 'state'],
#    prefix=['zone', 'state']
#)'

# Binary encode gender (map to 0/1)
df_cbs_customers['gender_binary'] = df_cbs_customers['gender'].map({
    'M': 1,
    'F': 0
})
df_cbs_customers.drop(columns=['gender', 'zone', 'state', 'age', 'age_category'], inplace=True)

print(df_cbs_customers.head())

#%% 5. Save the enriched CBS
output_dir = os.path.join(project_root, "data", "processed")
os.makedirs(output_dir, exist_ok=True)
cbs_merge_path = os.path.join(output_dir, "cdnow_cbs_customers.csv")
df_cbs_customers.to_csv(cbs_merge_path, index=False)
print(f"Saved CBS with customer demographics to {cbs_merge_path}")

# %%