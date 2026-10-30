import sys
import os
from pathlib import Path
import pandas as pd
import logging

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))

from data_provider import DataProvider, CACHE_DIR

logging.basicConfig(level=logging.INFO)

def fetch_and_save_nifty_5m():
    print("[INFO] Downloading / Updating Nifty 5m data...")
    dp = DataProvider()
    
    # Try fetching via DataProvider (Kite with yfinance fallback)
    # Kite allows max 60-100 days for 5m intraday. Let's ask for 60 days.
    df_5m = dp.fetch_intraday_5m("^NSEI", days=60, use_cache=False)
    
    if df_5m is not None and not df_5m.empty:
        out_path = CACHE_DIR / "NSEI_5m_historical.parquet"
        df_5m.to_parquet(out_path, index=False)
        print(f"[SUCCESS] Saved {len(df_5m)} 5-minute candles to {out_path}")
        print(f"   Date range: {df_5m['timestamps'].min()} to {df_5m['timestamps'].max()}")
        return df_5m
    else:
        print("[ERROR] Failed to fetch Nifty 5m data.")
        return None

if __name__ == "__main__":
    fetch_and_save_nifty_5m()
