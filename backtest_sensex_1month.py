import sys
from pathlib import Path
from datetime import time as dtime
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

from nifty_direction_engine.hybrid_engine import HybridSignalEngine

CACHE_PATH = CUR_DIR / "data_cache" / "sensex_5m_1month.parquet"

def run_sensex_1month_test():
    print("=" * 125)
    print("  📊  SENSEX 1-MONTH INTRADAY PREDICTION & VERIFICATION (10:00 AM -> 11:30 AM)")
    print("  Architecture: Hybrid (Kronos Foundation + LSTM + VWAP Acceptance Gate)")
    print("  Data Source : Zerodha Kite Live Feed (^BSESN / SENSEX)")
    print("=" * 125 + "\n")

    if not CACHE_PATH.exists():
        print(f"[ERR] Cache file not found: {CACHE_PATH}")
        return

    df = pd.read_parquet(CACHE_PATH).sort_values("timestamps").reset_index(drop=True)
    unique_days = sorted(list(set(df["timestamps"].dt.date)))
    print(f"[INFO] Total Sessions Captured: {len(unique_days)} ({len(df)} 5m candles)\n")

    # Initialize Engine
    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)

    print(f"\n{'DATE':<11} | {'10AM SPOT':<10} | {'PREDICTED':<12} | {'KRONOS':<8} | {'LSTM':<8} | {'11:30 CLOSE':<11} | {'ACTUAL MOVE':<13} | {'VERDICT':<8} | {'DETAILS / GATE'}")
    print("-" * 125)

    lookback = 75
    total_sessions = 0
    emitted = 0
    hits = 0
    misses = 0
    filtered = 0

    for day in unique_days:
        day_df = df[df["timestamps"].dt.date == day]
        morn = day_df[day_df["timestamps"].dt.time <= dtime(10, 0)]
        target_win = day_df[(day_df["timestamps"].dt.time > dtime(10, 0)) & (day_df["timestamps"].dt.time <= dtime(11, 30))]

        if len(morn) < 5 or len(target_win) < 5:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        total_sessions += 1
        slice_df = df.loc[cutoff_idx - lookback + 1:cutoff_idx].copy().reset_index(drop=True)

        spot_10am = float(morn["close"].iloc[-1])
        actual_1130 = float(target_win["close"].iloc[-1])
        actual_diff = actual_1130 - spot_10am
        actual_pct = (actual_diff / spot_10am) * 100.0

        # Generate Signal at 10:00 AM
        signal = engine.generate_signal(slice_df, deadband=0.0015)
        bias = signal.get("final_bias", "SIDEWAYS_NO_TRADE")
        reason = signal.get("reason", "")
        kronos_ret = signal.get("kronos_ret", 0.0) * 100.0
        lstm_ret = signal.get("lstm_ret", 0.0) * 100.0

        if bias == "SIDEWAYS_NO_TRADE":
            filtered += 1
            print(f"{str(day):<11} | {spot_10am:<10.1f} | {'NO_TRADE':<12} | {kronos_ret:>+6.2f}% | {lstm_ret:>+6.2f}% | {actual_1130:<11.1f} | {actual_diff:>+6.1f} ({actual_pct:>+5.2f}%) | {'GATED':<8} | {reason}")
            continue

        emitted += 1
        is_hit = False
        if bias == "BULLISH" and actual_pct > 0.0:
            is_hit = True
        elif bias == "BEARISH" and actual_pct < 0.0:
            is_hit = True

        if is_hit:
            hits += 1
            verdict = "HIT"
        else:
            misses += 1
            verdict = "MISS"

        print(f"{str(day):<11} | {spot_10am:<10.1f} | {bias:<12} | {kronos_ret:>+6.2f}% | {lstm_ret:>+6.2f}% | {actual_1130:<11.1f} | {actual_diff:>+6.1f} ({actual_pct:>+5.2f}%) | {verdict:<8} | {reason}")

    print("-" * 125)
    print("\n" + "=" * 50)
    print("      SENSEX 1-MONTH PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Total Sessions Evaluated : {total_sessions}")
    print(f"  Sessions Filtered (Gated): {filtered} ({filtered/max(1, total_sessions)*100:.1f}%)")
    print(f"  Trades Emitted           : {emitted}")
    print(f"  Hits (Correct Direction) : {hits}")
    print(f"  Misses                   : {misses}")
    if emitted > 0:
        win_rate = (hits / emitted) * 100.0
        print(f"  Directional Win Rate     : {win_rate:.2f}% ({hits}/{emitted})")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_sensex_1month_test()
