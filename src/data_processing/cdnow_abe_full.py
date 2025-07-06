"""
Filter cdnow_cbs_full.csv to only those customers appearing in cdnowElog.csv.
"""

import pandas as pd
import os
import sys

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

def main():
    # Paths to your inputs
    cbs_path   = os.path.join(project_root, "data", "processed", "cdnow_cbs_full.csv")
    elog_path  = os.path.join(project_root, "data", "processed", "cdnowElog.csv")
    output_path = os.path.join(project_root, "data", "processed", "cdnow_cbs_full_filtered.csv")

    # 1. Read in the data
    cbs  = pd.read_csv(cbs_path, dtype={"cust": str})
    elog = pd.read_csv(elog_path, dtype={"cust": str})

    # 2. Identify customers present in the event log
    valid_custs = set(elog["cust"].unique())

    # 3. Filter the CBS table
    filtered_cbs = cbs[cbs["cust"].isin(valid_custs)].copy()

    # 4. (Optional) Report how many were dropped
    total_before = len(cbs)
    total_after  = len(filtered_cbs)
    print(f"Dropped {total_before - total_after} rows; kept {total_after} of {total_before}.")

    # 5. Write out the filtered table
    filtered_cbs.to_csv(output_path, index=False)
    print(f"Filtered CBS saved to: {output_path}")

if __name__ == "__main__":
    main()