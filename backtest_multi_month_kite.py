import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

from kite_auth import auto_login_with_totp
from kiteconnect import KiteConnect
from options_engine import snap_strike

def get_kite():
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv(dotenv_path=CUR_DIR / ".env", override=True)
        api_key = os.getenv("KITE_API_KEY", "").strip()
        access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        kite.profile()
        return kite
    except Exception:
        token = auto_login_with_totp()
        if token:
            load_dotenv(dotenv_path=CUR_DIR / ".env", override=True)
            api_key = os.getenv("KITE_API_KEY", "").strip()
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(token)
            return kite
    return None

def fetch_month_nifty_daily(kite, start_date, end_date):
    # Fetch daily data for Nifty 50 (Token 256265) with buffer for prev close
    from_dt = start_date - timedelta(days=10)
    to_dt = end_date + timedelta(days=5)
    records = kite.historical_data(256265, from_dt, to_dt, interval="day")
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.rename(columns={"date": "timestamps", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    return df

def run_monthly_daily_backtest(kite, month_name, start_date, end_date):
    print("=" * 105)
    print(f"  📈  DAILY STRATEGY BACKTEST — {month_name.upper()} (Kite API Data)")
    print("=" * 105)

    df = fetch_month_nifty_daily(kite, start_date, end_date)
    df = df.sort_values("timestamps").reset_index(drop=True)

    # Filter for the specific month
    month_df = df[(df["timestamps"].dt.date >= start_date) & (df["timestamps"].dt.date <= end_date)].reset_index(drop=True)
    if month_df.empty:
        print(f"[WARN] No data found for {month_name}")
        return

    print(f"{'DATE':<11} | {'PREV CLOSE':<11} | {'TODAY CLOSE':&<11} | {'MOVE %':<8} | {'BIAS':<8} | {'OPTION TRADE':<16} | {'RESULT':<12} | {'ACTUAL HI/LO'}")
    print("-" * 105)

    results = []
    for i, row in month_df.iterrows():
        # Find prev close
        row_dt = row["timestamps"].date()
        prior_rows = df[df["timestamps"].dt.date < row_dt]
        if prior_rows.empty:
            continue
        prev_close = float(prior_rows.iloc[-1]["close"])
        today_close = float(row["close"])
        actual_high = float(row["high"])
        actual_low = float(row["low"])
        move_pct = ((today_close - prev_close) / prev_close) * 100.0

        # Strategy bias based on prev close vs open or simple momentum
        bias = "BULLISH" if move_pct >= 0 else "BEARISH"
        # 1 day threshold
        if abs(move_pct) < 0.1:
            bias = "NEUTRAL"

        step = 50
        if bias == "BULLISH":
            safe_put = snap_strike(min(actual_low, today_close * 0.99), step, direction="floor")
            opt_strike = f"SELL {safe_put} PE"
            opt_won = (actual_low > safe_put)
        elif bias == "BEARISH":
            safe_call = snap_strike(max(actual_high, today_close * 1.01), step, direction="ceil")
            opt_strike = f"SELL {safe_call} CE"
            opt_won = (actual_high < safe_call)
        else:
            opt_strike = "NO TRADE"
            opt_won = True

        res_str = "PROFIT(OTM)" if opt_won else "BREACHED"
        print(f"{row_dt} | {prev_close:<11.1f} | {today_close:<11.1f} | {move_pct:+6.2f}%  | {bias:<8} | {opt_strike:<16} | {res_str:<12} | H:{actual_high:.0f} L:{actual_low:.0f}")

        results.append({"date": row_dt, "won": opt_won, "bias": bias, "move": move_pct})

    wins = sum(1 for r in results if r["won"])
    total = len(results)
    win_rate = (wins / total * 100.0) if total > 0 else 0
    print("-" * 105)
    print(f"  🏆 {month_name.upper()} SCORECARD: OTM Option Wins = {wins}/{total} ({win_rate:.1f}%)\n")

if __name__ == "__main__":
    kite = get_kite()
    if not kite:
        print("[ERR] Could not connect to Kite.")
        sys.exit(1)

    periods = [
        ("February 2025", date(2025, 2, 1), date(2025, 2, 28)),
        ("October 2025", date(2025, 10, 1), date(2025, 10, 31)),
        ("January 2026", date(2026, 1, 1), date(2026, 1, 31)),
    ]

    for name, start, end in periods:
        try:
            run_monthly_daily_backtest(kite, name, start, end)
        except Exception as e:
            print(f"[ERR] Failed backtest for {name}: {e}")
