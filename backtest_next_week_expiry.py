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

# Black-Scholes Formula for Option Pricing & Decay Simulation
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


def run_next_week_expiry_backtest():
    print("=" * 135)
    print("  📊  NIFTY 50 1-MONTH TEST: NEXT WEEK EXPIRY (4-OTM / 200-PT BUFFER)")
    print("  Comparison: Next-Week Expiry (DTE ~8 to 12 Days) vs 10:00 AM -> 11:30 AM Window")
    print("  Strategy: Bearish -> Sell 4-OTM CE | Bullish -> Sell 4-OTM PE | Sideways -> Sell 4-OTM Strangle")
    print("=" * 135 + "\n")

    if not CACHE_PATH.exists():
        print(f"[ERR] Cache file not found: {CACHE_PATH}")
        return

    df = pd.read_parquet(CACHE_PATH).sort_values("timestamps").reset_index(drop=True)
    unique_days = sorted(list(set(df["timestamps"].dt.date)))
    print(f"[INFO] Loaded {len(unique_days)} sessions ({len(df)} 5m candles)\n")

    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)

    print(f"{'DATE':<11} | {'10AM SPOT':<10} | {'BIAS':<9} | {'SOLD STRIKE(S)':<22} | {'10:00 PREM':<12} | {'11:30 PREM':<12} | {'PNL (PTS)':<10} | {'11:30 SPOT':<10} | {'RESULT':<8} | {'STATUS'}")
    print("-" * 135)

    lookback = 75
    total_trades = 0
    wins = 0
    losses = 0
    total_points = 0.0

    r = 0.07  # Risk-free rate
    sigma = 0.135  # Historical realized / IV for Nifty (13.5%)

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
        high_in_win = float(target_win["high"].max())
        low_in_win = float(target_win["low"].min())

        atm_strike = int(round(spot_10am / 50.0) * 50)

        # Generate Signal at 10:00 AM
        signal = engine.generate_signal(slice_df, deadband=0.0015)
        bias = signal.get("final_bias", "SIDEWAYS_NO_TRADE")

        # Next week expiry typically has between 8 and 12 days to expiry (DTE)
        # Entry at 10:00 AM: t0 = 10.0 days; Exit at 11:30 AM (90 mins later): dt = 1.5 hours = 0.0625 trading days
        dte_entry = 10.0
        dte_exit = dte_entry - (1.5 / 6.0) # ~0.25 day of decay

        if bias == "BEARISH":
            sold_strike = atm_strike + 200
            p0 = bs_price(spot_10am, sold_strike, dte_entry, r, sigma, 'CE')
            p1 = bs_price(actual_1130, sold_strike, dte_exit, r, sigma, 'CE')
            pts = p0 - p1  # Option selling: profit = entry - exit
            strike_str = f"SELL {sold_strike} CE"
            breach = (high_in_win >= sold_strike)

        elif bias == "BULLISH":
            sold_strike = atm_strike - 200
            p0 = bs_price(spot_10am, sold_strike, dte_entry, r, sigma, 'PE')
            p1 = bs_price(actual_1130, sold_strike, dte_exit, r, sigma, 'PE')
            pts = p0 - p1
            strike_str = f"SELL {sold_strike} PE"
            breach = (low_in_win <= sold_strike)

        else:  # SIDEWAYS
            bias = "SIDEWAYS"
            ce_stk = atm_strike + 200
            pe_stk = atm_strike - 200
            p0_ce = bs_price(spot_10am, ce_stk, dte_entry, r, sigma, 'CE')
            p1_ce = bs_price(actual_1130, ce_stk, dte_exit, r, sigma, 'CE')
            p0_pe = bs_price(spot_10am, pe_stk, dte_entry, r, sigma, 'PE')
            p1_pe = bs_price(actual_1130, pe_stk, dte_exit, r, sigma, 'PE')
            p0 = p0_ce + p0_pe
            p1 = p1_ce + p1_pe
            pts = p0 - p1
            strike_str = f"{pe_stk}PE / {ce_stk}CE"
            breach = (high_in_win >= ce_stk) or (low_in_win <= pe_stk)

        total_trades += 1
        total_points += pts

        if not breach and pts >= -0.5:
            wins += 1
            verdict = "WIN ✅"
            status = "Decay Captured"
        else:
            losses += 1
            verdict = "LOSS ❌"
            status = "Adverse Spot Move"

        print(f"{str(day):<11} | {spot_10am:<10.1f} | {bias:<9} | {strike_str:<22} | ₹{p0:>9.2f}  | ₹{p1:>9.2f}  | {pts:>+7.2f} pt | {actual_1130:<10.1f} | {verdict:<8} | {status}")

    print("-" * 135)
    print("\n" + "=" * 55)
    print("      NEXT-WEEK EXPIRY PERFORMANCE SUMMARY")
    print("=" * 55)
    print(f"  Total Trading Sessions   : {total_trades}")
    print(f"  Profitable Sessions      : {wins}")
    print(f"  Losing Sessions          : {losses}")
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100.0
        print(f"  Win Rate                 : {win_rate:.2f}% ({wins}/{total_trades})")
        print(f"  Net Total Points Gained  : {total_points:+.2f} pts")
        print(f"  Average Points Per Day   : {total_points / total_trades:+.2f} pts")
        print(f"  Approx Net PnL (1 Lot)   : ₹{total_points * 65:+,.2f}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    run_next_week_expiry_backtest()
