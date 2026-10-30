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

CACHE_PATH = CUR_DIR / "data_cache" / "NSEI_5m_historical.parquet"

def run_otm_option_selling_test():
    print("=" * 125)
    print("  📊  NIFTY 50 1-MONTH 4-OTM OPTION SELLING TEST (10:00 AM -> 11:30 AM)")
    print("  Strategy: If BEARISH -> SELL 4-OTM Call (CE). If BULLISH -> SELL 4-OTM Put (PE).")
    print("  4-OTM = 4 strikes away (4 x 50 = 200 points OTM buffer)")
    print("  Win Condition: Spot at 11:30 AM stays strictly OUT of the money (never breaches sold strike)")
    print("=" * 125 + "\n")

    if not CACHE_PATH.exists():
        print(f"[ERR] Cache file not found: {CACHE_PATH}")
        return

    df = pd.read_parquet(CACHE_PATH).sort_values("timestamps").reset_index(drop=True)
    unique_days = sorted(list(set(df["timestamps"].dt.date)))
    print(f"[INFO] Loaded {len(unique_days)} sessions ({len(df)} 5m candles)\n")

    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)

    print(f"{'DATE':<11} | {'10AM SPOT':<10} | {'PREDICTION':<10} | {'SOLD OPTION':<16} | {'BUFFER':<8} | {'11:30 SPOT':<10} | {'MAX DRIFT':<15} | {'RESULT':<8} | {'PNL STATUS'}")
    print("-" * 125)

    lookback = 75
    total_trades = 0
    wins = 0
    losses = 0
    gated_out = 0

    for day in unique_days:
        day_df = df[df["timestamps"].dt.date == day]
        morn = day_df[day_df["timestamps"].dt.time <= dtime(10, 0)]
        target_win = day_df[(day_df["timestamps"].dt.time > dtime(10, 0)) & (day_df["timestamps"].dt.time <= dtime(11, 30))]

        if len(morn) < 5 or len(target_win) < 5:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        slice_df = df.loc[cutoff_idx - lookback + 1:cutoff_idx].copy().reset_index(drop=True)
        spot_10am = float(morn["close"].iloc[-1])
        actual_1130 = float(target_win["close"].iloc[-1])
        high_in_window = float(target_win["high"].max())
        low_in_window = float(target_win["low"].min())

        atm_strike = int(round(spot_10am / 50.0) * 50)

        # Generate Signal
        signal = engine.generate_signal(slice_df, deadband=0.0015)
        bias = signal.get("final_bias", "SIDEWAYS_NO_TRADE")

        if bias == "SIDEWAYS_NO_TRADE":
            # For sideways, sell BOTH 4-OTM Call and 4-OTM Put (Strangle)
            sold_ce = atm_strike + 200 # 4 strikes OTM CE
            sold_pe = atm_strike - 200 # 4 strikes OTM PE
            total_trades += 1

            breach_ce = high_in_window >= sold_ce
            breach_pe = low_in_window <= sold_pe

            if not breach_ce and not breach_pe:
                wins += 1
                verdict = "FULL WIN"
                drift_str = f"H:{high_in_window:.0f} L:{low_in_window:.0f}"
                pnl_str = "Decay Captured (100%)"
            else:
                losses += 1
                verdict = "BREACH"
                drift_str = f"H:{high_in_window:.0f} L:{low_in_window:.0f}"
                pnl_str = "Strangle Boundary Hit"

            print(f"{str(day):<11} | {spot_10am:<10.1f} | {'SIDEWAYS':<10} | {sold_pe}PE / {sold_ce}CE | ±200 pts | {actual_1130:<10.1f} | {drift_str:<15} | {verdict:<8} | {pnl_str}")

        elif bias == "BEARISH":
            # Sell 4-OTM CE (Call) -> Ceiling at atm_strike + 200
            sold_strike = atm_strike + 200
            total_trades += 1

            # Loss if index drifted up and touched or exceeded the sold Call strike
            breach = high_in_window >= sold_strike

            if not breach:
                wins += 1
                verdict = "FULL WIN"
                pnl_str = f"Expired OTM (+{(sold_strike - actual_1130):.1f} pts safe)"
            else:
                losses += 1
                verdict = "BREACH"
                pnl_str = f"Call Breached by {(high_in_window - sold_strike):.1f} pts"

            drift_str = f"High: {high_in_window:.1f}"
            print(f"{str(day):<11} | {spot_10am:<10.1f} | {'BEARISH':<10} | SELL {sold_strike} CE     | +200 pts | {actual_1130:<10.1f} | {drift_str:<15} | {verdict:<8} | {pnl_str}")

        elif bias == "BULLISH":
            # Sell 4-OTM PE (Put) -> Floor at atm_strike - 200
            sold_strike = atm_strike - 200
            total_trades += 1

            # Loss if index drifted down and touched or fell below the sold Put strike
            breach = low_in_window <= sold_strike

            if not breach:
                wins += 1
                verdict = "FULL WIN"
                pnl_str = f"Expired OTM (+{(actual_1130 - sold_strike):.1f} pts safe)"
            else:
                losses += 1
                verdict = "BREACH"
                pnl_str = f"Put Breached by {(sold_strike - low_in_window):.1f} pts"

            drift_str = f"Low: {low_in_window:.1f}"
            print(f"{str(day):<11} | {spot_10am:<10.1f} | {'BULLISH':<10} | SELL {sold_strike} PE     | -200 pts | {actual_1130:<10.1f} | {drift_str:<15} | {verdict:<8} | {pnl_str}")

    print("-" * 125)
    print("\n" + "=" * 50)
    print("      4-OTM OPTION SELLING AUDIT")
    print("=" * 50)
    print(f"  Total Trades Tested      : {total_trades}")
    print(f"  Full Wins (OTM Expired)  : {wins}")
    print(f"  Losses (Strike Breached) : {losses}")
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100.0
        print(f"  Option Seller Win Rate   : {win_rate:.2f}% ({wins}/{total_trades})")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_otm_option_selling_test()
