import pandas as pd
from datetime import datetime

def load_transactions(path: str) -> pd.DataFrame:
  df = pd.read_csv(path, encoding='ISO-8859-1')
  df.columns = [c.strip() for c in df.columns]
  df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
  return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
# Remove cancelled invoices
  df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
  # Remove invalid quantities and prices
  df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
# Drop missing customers
  df = df.dropna(subset=['CustomerID'])
  df['CustomerID'] = df['CustomerID'].astype(int)
  return df  