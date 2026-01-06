# app/utils.py
import pandas as pd
from io import BytesIO

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=True)
    buffer.seek(0)
    return buffer.getvalue()
