import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))

from data_provider import DataProvider, CACHE_DIR

logging.basicConfig(level=logging.INFO)

def fetch_chunked_nifty_data(days_total: int = 180):
    """
    Kite allows fetching up to 60-100 days of 5-minute candles per API call.
    To get 6 months (180 days), we can fetch in consecutive 50-day windows.
    """
    print(f"[INFO] Fetching {days_total} days of 5m Nifty candles from Kite...")
    dp = DataProvider()
    if not dp.kite:
        print("[ERR] Kite client not initialized.")
        return None

    token = dp.get_instrument_token("^NSEI")
    if not token:
        print("[ERR] Could not resolve token for ^NSEI")
        return None

    all_records = []
    chunk_size = 50
    now = datetime.now()

    for start_offset in range(0, days_total, chunk_size):
        end_offset = min(start_offset + chunk_size, days_total)
        to_date = now - timedelta(days=start_offset)
        from_date = now - timedelta(days=end_offset)

        print(f"   Fetching chunk: {from_date.strftime('%Y-%m-%d')} -> {to_date.strftime('%Y-%m-%d')}...")
        try:
            records = dp.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="5minute"
            )
            if records:
                all_records.extend(records)
                print(f"   Received {len(records)} candles.")
        except Exception as e:
            print(f"   [WARN] Chunk fetch failed: {e}")

    if not all_records:
        print("[ERR] No records fetched.")
        return None

    df = pd.DataFrame(all_records)
    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
    df = df.drop_duplicates(subset=["timestamps"]).sort_values("timestamps").reset_index(drop=True)
    df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()

    out_file = CACHE_DIR / "nifty_5m_6months.parquet"
    df.to_parquet(out_file, index=False)
    print(f"[SUCCESS] Downloaded and saved {len(df)} 5-minute candles to {out_file}")
    print(f"   Total Date Range: {df['timestamps'].min()} to {df['timestamps'].max()}")
    return df

if __name__ == "__main__":
    fetch_chunked_nifty_data(days_total=180)
