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

def bs_price(spot, strike, t_days, r, sigma, opt_type):
    from scipy.stats import norm
    t = max(t_days / 365.0, 1e-4)
    denom = sigma * np.sqrt(t)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * t) / denom
    d2 = d1 - denom
    if opt_type == 'CE':
        price = spot * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
    else:
        price = strike * np.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return max(price, 0.05)


def run_exit_time_comparison():
    print("=" * 135)
    print("  📊  NIFTY 50 1-MONTH TEST: 11:30 AM EXIT vs 3:00 PM EXIT (CURRENT-WEEK EXPIRY)")
    print("  Strategy: 4-OTM (200-Pt Buffer) Entered at 10:00 AM IST")
    print("  Scenario A: Quick Scalp Exit at 11:30 AM IST (1.5 Hours Holding)")
    print("  Scenario B: Full Session Hold Exit at 3:00 PM IST (5.0 Hours Holding)")
    print("=" * 135 + "\n")

    if not CACHE_PATH.exists():
        print(f"[ERR] Cache file not found: {CACHE_PATH}")
        return

    df = pd.read_parquet(CACHE_PATH).sort_values("timestamps").reset_index(drop=True)
    unique_days = sorted(list(set(df["timestamps"].dt.date)))
    print(f"[INFO] Loaded {len(unique_days)} sessions ({len(df)} 5m candles)\n")

    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)

    r = 0.07
    sigma = 0.135
    lookback = 75

    tot_1130_pts = 0.0
    tot_1500_pts = 0.0
    wins_1130 = 0
    wins_1500 = 0
    breaches_1130 = 0
    breaches_1500 = 0
    total_sessions = 0

    print(f"{'DATE':<11} | {'10AM SPOT':<10} | {'BIAS':<9} | {'11:30 SPOT':<10} | {'3:00PM SPOT':<11} | {'11:30 PNL':<14} | {'3:00PM PNL':<14} | {'WHICH EXIT WON?'}")
    print("-" * 135)

    for day in unique_days:
        day_df = df[df["timestamps"].dt.date == day]
        morn = day_df[day_df["timestamps"].dt.time <= dtime(10, 0)]
        target_1130 = day_df[(day_df["timestamps"].dt.time > dtime(10, 0)) & (day_df["timestamps"].dt.time <= dtime(11, 30))]
        target_1500 = day_df[(day_df["timestamps"].dt.time > dtime(10, 0)) & (day_df["timestamps"].dt.time <= dtime(15, 0))]

        if len(morn) < 5 or len(target_1130) < 5 or len(target_1500) < 20:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        slice_df = df.loc[cutoff_idx - lookback + 1:cutoff_idx].copy().reset_index(drop=True)
        p10 = float(morn["close"].iloc[-1])
        p1130 = float(target_1130["close"].iloc[-1])
        p1500 = float(target_1500["close"].iloc[-1])

        # Range excursions
        high_1130 = float(target_1130["high"].max())
        low_1130 = float(target_1130["low"].min())
        high_1500 = float(target_1500["high"].max())
        low_1500 = float(target_1500["low"].min())

        atm = int(round(p10 / 50.0) * 50)

        # Generate Signal at 10:00 AM
        signal = engine.generate_signal(slice_df, deadband=0.0015)
        bias = signal.get("final_bias", "SIDEWAYS_NO_TRADE")

        # DTE for current week: entry ~2.5 days
        dte_1000 = 2.5
        dte_1130 = dte_1000 - (1.5 / 6.0) # 90 mins = 0.25 day decay
        dte_1500 = dte_1000 - (5.0 / 6.0) # 5.0 hours = 0.833 day decay

        if bias == "BEARISH":
            stk = atm + 200
            p0 = bs_price(p10, stk, dte_1000, r, sigma, 'CE')
            pnl_1130 = p0 - bs_price(p1130, stk, dte_1130, r, sigma, 'CE')
            pnl_1500 = p0 - bs_price(p1500, stk, dte_1500, r, sigma, 'CE')
            b1130 = (high_1130 >= stk)
            b1500 = (high_1500 >= stk)

        elif bias == "BULLISH":
            stk = atm - 200
            p0 = bs_price(p10, stk, dte_1000, r, sigma, 'PE')
            pnl_1130 = p0 - bs_price(p1130, stk, dte_1130, r, sigma, 'PE')
            pnl_1500 = p0 - bs_price(p1500, stk, dte_1500, r, sigma, 'PE')
            b1130 = (low_1130 <= stk)
            b1500 = (low_1500 <= stk)

        else: # SIDEWAYS
            bias = "SIDEWAYS"
            stk_ce = atm + 200
            stk_pe = atm - 200
            p0 = bs_price(p10, stk_ce, dte_1000, r, sigma, 'CE') + bs_price(p10, stk_pe, dte_1000, r, sigma, 'PE')
            pnl_1130 = p0 - (bs_price(p1130, stk_ce, dte_1130, r, sigma, 'CE') + bs_price(p1130, stk_pe, dte_1130, r, sigma, 'PE'))
            pnl_1500 = p0 - (bs_price(p1500, stk_ce, dte_1500, r, sigma, 'CE') + bs_price(p1500, stk_pe, dte_1500, r, sigma, 'PE'))
            b1130 = (high_1130 >= stk_ce) or (low_1130 <= stk_pe)
            b1500 = (high_1500 >= stk_ce) or (low_1500 <= stk_pe)

        total_sessions += 1
        tot_1130_pts += pnl_1130
        tot_1500_pts += pnl_1500

        if pnl_1130 > 0: wins_1130 += 1
        if pnl_1500 > 0: wins_1500 += 1
        if b1130: breaches_1130 += 1
        if b1500: breaches_1500 += 1

        winner = "3:00 PM 💰" if pnl_1500 > pnl_1130 else "11:30 AM ⏱️"
        print(f"{str(day):<11} | {p10:<10.1f} | {bias:<9} | {p1130:<10.1f} | {p1500:<11.1f} | {pnl_1130:>+7.2f} pts     | {pnl_1500:>+7.2f} pts     | {winner}")

    print("-" * 135)
    print("\n" + "=" * 65)
    print("      🏆 EXIT TIME COMPARISON SUMMARY (38 SESSIONS)")
    print("=" * 65)
    print(f"  Metric                     11:30 AM Exit       3:00 PM Exit")
    print("-" * 65)
    print(f"  Profitable Sessions        {wins_1130}/{total_sessions} ({wins_1130/total_sessions*100:.1f}%)     {wins_1500}/{total_sessions} ({wins_1500/total_sessions*100:.1f}%)")
    print(f"  4-OTM Strikes Breached     {breaches_1130}/{total_sessions} ({breaches_1130/total_sessions*100:.1f}%)      {breaches_1500}/{total_sessions} ({breaches_1500/total_sessions*100:.1f}%)")
    print(f"  Net Total Points Gained    {tot_1130_pts:>+8.2f} pts       {tot_1500_pts:>+8.2f} pts")
    print(f"  Average Points Per Day     {tot_1130_pts/total_sessions:>+8.2f} pts       {tot_1500_pts/total_sessions:>+8.2f} pts")
    print(f"  Approx Net PnL (1 Lot)     ₹{tot_1130_pts*65:>+8,.2f}        ₹{tot_1500_pts*65:>+8,.2f}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_exit_time_comparison()
