# Add pandas and numpy imports for _fmt
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Helper: enforce uniform decimal display (e.g. 0.63, 2.57, …)
# ---------------------------------------------------------------------
def _fmt(df: pd.DataFrame, dec: int) -> pd.DataFrame:
    """Return a copy of *df* with all float cells formatted to *dec* decimals."""
    fmt = f"{{:.{dec}f}}".format
    return df.applymap(lambda v: fmt(v) if isinstance(v, (float, np.floating)) else v)
# ------------------------------------------------------------------


