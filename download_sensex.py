import sys
from pathlib import Path
from datetime import datetime, timedelta
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

def fetch_sensex_5m():
    print("[INFO] Fetching 35 days of 5m Sensex candles from Kite / Yahoo...")
    dp = DataProvider()
    
    # Try fetching via Kite or fallback to yfinance
    df = dp.fetch_intraday_5m("^BSESN", days=35, use_cache=False)
    if df is not None and not df.empty:
        out_path = CACHE_DIR / "sensex_5m_1month.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[SUCCESS] Downloaded and saved {len(df)} 5-minute candles to {out_path}")
        print(f"   Date Range: {df['timestamps'].min()} to {df['timestamps'].max()}")
        return df
    else:
        print("[ERR] Failed to download Sensex data.")
        return None

if __name__ == "__main__":
    fetch_sensex_5m()
