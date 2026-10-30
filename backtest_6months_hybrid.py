import sys
import os
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

CACHE_PATH = CUR_DIR / "data_cache" / "nifty_5m_6months.parquet"

def run_6month_hybrid_backtest():
    print("=" * 115)
    print("  HYBRID INTRADAY NIFTY 50 BACKTEST: KRONOS + TRAINED LSTM + VWAP & FLOW GATES")
    print("  Data Source : Zerodha Kite 5-Minute Historical Index Feed (Past 6 Months)")
    print("  Horizon     : 10:00 AM -> 11:30 AM IST (18 Candlestick Bars)")
    print("=" * 115 + "\n")

    if not CACHE_PATH.exists():
        print(f"[ERR] Cache file not found at {CACHE_PATH}")
        return

    df = pd.read_parquet(CACHE_PATH).sort_values("timestamps").reset_index(drop=True)
    unique_days = sorted(list(set(df["timestamps"].dt.date)))
    print(f"[INFO] Total Sessions: {len(unique_days)} ({len(df)} total 5m bars)")

    # Initialize the hybrid engine (Kronos + trained LSTM)
    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)

    lookback = 100
    total_evaluated = 0
    signals_emitted = 0
    hits = 0
    misses = 0
    gated_out = 0

    results = []

    print(f"\n{'DATE':<11} | {'10AM SPOT':<10} | {'PRED BIAS':<10} | {'KRONOS':<8} | {'LSTM':<8} | {'ENSEMBLE':<9} | {'11:30 CLOSE':<11} | {'ACTUAL %':<9} | {'RESULT':<8} | {'GATE STATUS / REASON'}")
    print("-" * 125)

    for idx in range(len(unique_days)):
        test_day = unique_days[idx]
        day_df = df[df["timestamps"].dt.date == test_day]
        morn = day_df[day_df["timestamps"].dt.time <= dtime(10, 0)]
        target_win = day_df[(day_df["timestamps"].dt.time > dtime(10, 0)) & (day_df["timestamps"].dt.time <= dtime(11, 30))]

        if len(morn) < 5 or len(target_win) < 5:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        total_evaluated += 1

        # Lookback slice of 100 bars
        slice_df = df.loc[cutoff_idx - lookback + 1:cutoff_idx].copy().reset_index(drop=True)
        spot_10am = float(morn["close"].iloc[-1])
        actual_1130 = float(target_win["close"].iloc[-1])
        actual_ret_pct = ((actual_1130 - spot_10am) / spot_10am) * 100.0

        signal = engine.generate_signal(slice_df, deadband=0.0015)
        bias = signal.get("final_bias", "SIDEWAYS_NO_TRADE")
        reason = signal.get("reason", "")
        kronos_ret = signal.get("kronos_ret", 0.0) * 100.0
        lstm_ret = signal.get("lstm_ret", 0.0) * 100.0
        ensemble_ret = signal.get("pred_return", 0.0) * 100.0

        if bias == "SIDEWAYS_NO_TRADE":
            gated_out += 1
            print(f"{str(test_day):<11} | {spot_10am:<10.1f} | {'FILTERED':<10} | {kronos_ret:>+6.2f}% | {lstm_ret:>+6.2f}% | {ensemble_ret:>+7.2f}% | {'-':<11} | {actual_ret_pct:>+7.2f}% | {'REJECT':<8} | {reason}")
            continue

        signals_emitted += 1
        is_hit = False
        if bias == "BULLISH" and actual_ret_pct > 0.0:
            is_hit = True
        elif bias == "BEARISH" and actual_ret_pct < 0.0:
            is_hit = True

        if is_hit:
            hits += 1
            res_label = "HIT"
        else:
            misses += 1
            res_label = "MISS"

        print(f"{str(test_day):<11} | {spot_10am:<10.1f} | {bias:<10} | {kronos_ret:>+6.2f}% | {lstm_ret:>+6.2f}% | {ensemble_ret:>+7.2f}% | {actual_1130:<11.1f} | {actual_ret_pct:>+7.2f}% | {res_label:<8} | {reason}")

    print("-" * 125)
    print("\n" + "=" * 50)
    print("      HYBRID ENGINE PERFORMANCE AUDIT")
    print("=" * 50)
    print(f"  Sessions Evaluated    : {total_evaluated}")
    print(f"  Signals Filtered (Gated): {gated_out} ({gated_out/max(1, total_evaluated)*100:.1f}% Rejection Rate)")
    print(f"  Signals Emitted       : {signals_emitted}")
    print(f"  Hits                  : {hits}")
    print(f"  Misses                : {misses}")
    if signals_emitted > 0:
        win_rate = (hits / signals_emitted) * 100.0
        print(f"  Directional Win Rate  : {win_rate:.2f}% ({hits}/{signals_emitted})")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_6month_hybrid_backtest()
