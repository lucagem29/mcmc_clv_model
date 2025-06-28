"""
Convert an event log into a customer-by-sufficient-statistic (CBS) DataFrame.

This function processes a transaction event log to produce a summary
DataFrame with one row per customer, suitable for estimating model parameters.
The time units for the summary statistics are configurable, and the function
supports splitting data into calibration and holdout periods.

Transactions with identical customer and date are aggregated by summing sales.
"""

import pandas as pd
import numpy as np

# Convert Event Log to customer-level summary statistic
#
# Efficient implementation for the conversion of an event log into a
# customer-by-sufficient-statistic (CBS) DataFrame, with a row for each
# customer, which is the required data format for estimating model parameters.
#
# The time unit for expressing `t_x`, `T_cal` and `litt` are
# determined via the argument `units`, which is passed to NumPy timedelta.
#
# Argument `T_tot` allows one to specify the end of the observation period.
# Any event that occurs after that date is discarded.
#
# Argument `T_cal` allows one to split the summary statistics into a
# calibration and a holdout period. If `T_cal` is not provided,
# the entire observation period is used for estimating model parameters.
#
# Transactions with identical `cust` and `date` are treated as a
# single transaction, with `sales` being summed up.
def elog2cbs(elog, units="week", T_cal=None, T_tot=None):
    if not isinstance(elog, pd.DataFrame):
        raise ValueError("elog must be a pandas DataFrame")
    if 'cust' not in elog.columns or 'date' not in elog.columns:
        raise ValueError("elog must contain 'cust' and 'date' columns")
    if elog.empty:
        return pd.DataFrame(columns=["cust", "x", "t.x", "litt", "first", "T.cal"])

    elog = elog.copy()
    elog['date'] = pd.to_datetime(elog['date'])
    if 'sales' not in elog.columns:
        elog['sales'] = 1
    else:
        if not pd.api.types.is_numeric_dtype(elog['sales']):
            raise ValueError("'sales' column must be numeric")

    if T_cal is None:
        T_cal = elog['date'].max()
    else:
        T_cal = pd.to_datetime(T_cal)

    if T_tot is None:
        T_tot = elog['date'].max()
    else:
        T_tot = pd.to_datetime(T_tot)

    has_holdout = T_cal < T_tot

    # Merge multiple transactions on same date
    elog = elog.groupby(['cust', 'date'], as_index=False).agg({'sales': 'sum'})

    # Calculate intertransaction times
    elog = elog.sort_values(['cust', 'date'])
    elog['first'] = elog.groupby('cust')['date'].transform('min')
    t = (elog['date'] - elog['first']) / np.timedelta64(1, units)
    elog['t'] = t
    elog['itt'] = elog.groupby('cust')['t'].diff().fillna(0)

    # Count events and compute stats in the calibration period
    # Calibration period
    cal = elog[elog['date'] <= T_cal]
    cal_stats = cal.groupby('cust').agg(
        x=('date', lambda d: len(d) - 1),
        t_x=('t', 'max'),
        litt=('itt', lambda x: np.log(x[x > 0]).sum()),
        sales=('sales', 'sum'),
        sales_x=('sales', lambda s: s.iloc[1:].sum() if len(s) > 1 else 0),
        first=('first', 'first')
    ).reset_index()

    cal_stats['T_cal'] = (T_cal - cal_stats['first']) / np.timedelta64(1, units)

    # Count events and compute stats in the holdout (validation) period
    # Holdout period
    if has_holdout:
        cal_stats['T_star'] = (T_tot - cal_stats['first']) / np.timedelta64(1, units) - cal_stats['T_cal']
        val = elog[(elog['date'] > T_cal) & (elog['date'] <= T_tot)]
        val_stats = val.groupby('cust').agg(x_star=('date', 'count'), sales_star=('sales', 'sum')).reset_index()
        cal_stats = pd.merge(cal_stats, val_stats, on='cust', how='left').fillna({'x_star': 0, 'sales_star': 0})

    # Return the final customer-by-sufficient-statistic DataFrame
    return cal_stats