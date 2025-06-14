import pandas as pd
import numpy as np
from datetime import datetime

# Load the CDNow dataset
# Note: You'll need to have the cdnowElog.txt file in your data directory
def load_cdnow_data(file_path):
    """
    Load and preprocess the CDNow dataset from a text file.
    
    Parameters:
    -----------
    file_path : str
        Path to the cdnowElog.txt file
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed CDNow dataset
    """
    # Read the text file with space or tab delimiter
    df = pd.read_csv(file_path, 
                     sep='\s+',  # This will handle both space and tab delimiters
                     names=['cust', 'sampleid', 'date', 'cds', 'sales'],
                     dtype={'cust': int, 'sampleid': int, 'date': str, 'cds': int, 'sales': float})
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    return df

def create_customer_summary(df, T_cal='1997-09-30', T_tot='1998-06-30'):
    """
    Convert event log to customer-by-sufficient-statistic summary.
    Split into calibration and holdout periods.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Event log dataframe
    T_cal : str
        Calibration period end date
    T_tot : str
        Total period end date
        
    Returns:
    --------
    pd.DataFrame
        Customer-by-sufficient-statistic summary with columns:
        - x: Number of repeat purchases in calibration period
        - t.x: Recency (weeks since first purchase)
        - T.cal: Length of calibration period (weeks)
        - x.star: Number of purchases in holdout period
        - T.star: Length of holdout period (weeks)
        - first.sales: Amount of first purchase (in thousands)
    """
    # Convert dates to datetime
    T_cal = pd.to_datetime(T_cal)
    T_tot = pd.to_datetime(T_tot)
    
    # Get the start date of the study (first purchase across all customers)
    study_start = df['date'].min()
    
    # Calculate customer statistics
    cbs = df.groupby('cust').agg({
        'date': lambda x: (x <= T_cal).sum() - 1,  # x (frequency, excluding first purchase)
    }).rename(columns={'date': 'x'})
    
    # Calculate recency (in weeks)
    first_date = df.groupby('cust')['date'].min()
    last_date = df[df['date'] <= T_cal].groupby('cust')['date'].max()
    cbs['t.x'] = ((last_date - first_date).dt.days / 7).fillna(0)
    
    # Calculate T.cal (in weeks) - same for all customers
    cbs['T.cal'] = ((T_cal - study_start).days / 7)
    
    # Calculate holdout period statistics
    holdout = df[(df['date'] > T_cal) & (df['date'] <= T_tot)]
    # Compute holdout stats: number of purchases in holdout (x.star)
    holdout_stats = holdout.groupby('cust').agg({'sales': 'count'}).rename(columns={'sales': 'x.star'})
    cbs = cbs.merge(holdout_stats, left_index=True, right_index=True, how='left')
    cbs['x.star'] = cbs['x.star'].fillna(0)
    # Calculate T.star (in weeks): derived from T_tot and T_cal, not hardcoded
    cbs['T.star'] = ((T_tot - T_cal).days) / 7
    
    # Add first purchase amount as covariate
    # Use R logic: first.sales is amount of first purchase in thousands
    first_sales = df.sort_values('date').groupby('cust').first().sales * 1e-3
    cbs['first.sales'] = first_sales
    
    return cbs

if __name__ == "__main__":
    # Example usage
    # Note: You'll need to provide the correct path to your cdnowElog.txt file
    file_path = "cdnowElog.txt"
    
    # Load and preprocess data
    cdnow_elog = load_cdnow_data(file_path)
    print("Date range:", cdnow_elog['date'].min(), "to", cdnow_elog['date'].max())
    
    # Create customer summary
    cbs = create_customer_summary(cdnow_elog)
    print("\nCustomer summary statistics:")
    print(cbs.head())
    print(cbs.info())
    
    # Save to CSV
    output_file = "cdnow_cbs.csv"
    cbs.to_csv(output_file, index=True)
    print(f"\nCustomer summary saved to {output_file}") 