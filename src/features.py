import pandas as pd
import numpy as np
from datetime import datetime

def compute_rfm(df: pd.DataFrame, snapshot_date: datetime = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) and additional features for each customer.

    Args:
        df (pd.DataFrame): Transactions dataframe with columns ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']
        snapshot_date (datetime, optional): Reference date for Recency calculation. Defaults to max InvoiceDate + 1 day.

    Returns:
        pd.DataFrame: RFM table with additional metrics.
    """
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Create TotalPrice column for easier aggregation
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Aggregate RFM
    grouped = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),  # number of distinct invoices
        Monetary=('TotalPrice', 'sum'),
        TotalQuantity=('Quantity', 'sum')
    ).reset_index()

    # Clip Monetary to avoid zero
    grouped['Monetary'] = grouped['Monetary'].clip(lower=0.01)

    # Additional metrics
    grouped['AvgOrderValue'] = grouped['Monetary'] / grouped['Frequency']
    grouped['AvgItemsPerOrder'] = grouped['TotalQuantity'] / grouped['Frequency']

    return grouped
