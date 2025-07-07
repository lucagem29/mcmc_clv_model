# %% 1. Import libraries & set up project root
# -- 1. Import libraries & set up project root --
import requests, io
import os
import sys
import pandas as pd
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

# %% 2. Download and save data from Hugging Face
# -- 2. Download and save data from Hugging Face --

base = "https://huggingface.co/datasets/ZennyKenny/CDNOW/resolve/main/"
files = ["purchases.csv", "customers.csv"]

dfs = {}
for fn in files:
    r = requests.get(base + fn)
    r.raise_for_status()
    dfs[fn] = pd.read_csv(io.StringIO(r.text), parse_dates=["date"] if fn=="purchases.csv" else None)

df_purchases = dfs["purchases.csv"]
df_customers = dfs["customers.csv"]
# Rename id to cust for consistency with CBS data
df_customers.rename(columns={"id": "cust"}, inplace=True)
df_purchases.rename(columns={"users_id": "cust", "amt": "sales"}, inplace=True)

df_customers.describe()
df_purchases.describe()

# Save customer characteristics df locally
output_dir = os.path.join(project_root, "data", "raw")
os.makedirs(output_dir, exist_ok=True)
df_customers.to_csv(os.path.join(output_dir, "cdnow_fullCovar.csv"), index=False)
df_purchases.to_csv(os.path.join(output_dir, "cdnow_fullElog.csv"), index=False)

# %%
