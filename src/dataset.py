# src/dataset.py
from __future__ import annotations

import pandas as pd

TARGET_COL = "default payment next month"


def load_uci_credit_xls(xls_path: str) -> pd.DataFrame:
    """
    Loads UCI 'Default of credit card clients' dataset from the XLS file.

    The XLS has a non-standard format:
    - Row 0 is a title/description line.
    - Row 1 contains the real column headers.
    - Data starts from row 2.
    """
    # Read with header on row 1 (0-indexed), so pandas uses row 1 as columns
    df = pd.read_excel(xls_path, header=1)

    # Defensive cleanup: strip whitespace and normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop completely empty rows (sometimes appear)
    df = df.dropna(how="all")

    # Sometimes an extra header row still sneaks in; remove rows where target == column name
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL].astype(str).str.lower().ne(TARGET_COL.lower())]
    else:
        raise KeyError(f"Expected target column '{TARGET_COL}' not found. Columns={list(df.columns)}")

    # Convert target to int
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int)

    return df
