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


def run_comparison_current_vs_next():
    df = pd.read_parquet(CACHE_PATH).sort_values("timestamps").reset_index(drop=True)
    unique_days = sorted(list(set(df["timestamps"].dt.date)))
    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)

    r = 0.07
    sigma = 0.135
    lookback = 75

    cur_pts_tot = 0.0
    next_pts_tot = 0.0
    cur_wins = 0
    next_wins = 0
    total = 0

    print("=" * 130)
    print(f"{'DATE':<11} | {'10AM SPOT':<10} | {'BIAS':<9} | {'11:30 SPOT':<10} | {'CURRENT-WK PNL':<16} | {'NEXT-WK PNL':<16} | {'WHICH WON HIGHER PnL?'}")
    print("-" * 130)

    for day in unique_days:
        day_df = df[df["timestamps"].dt.date == day]
        morn = day_df[day_df["timestamps"].dt.time <= dtime(10, 0)]
        target_win = day_df[(day_df["timestamps"].dt.time > dtime(10, 0)) & (day_df["timestamps"].dt.time <= dtime(11, 30))]

        if len(morn) < 5 or len(target_win) < 5: continue
        cutoff = morn.index[-1]
        if cutoff < lookback: continue

        slice_df = df.loc[cutoff - lookback + 1:cutoff].copy().reset_index(drop=True)
        p10 = float(morn["close"].iloc[-1])
        p1130 = float(target_win["close"].iloc[-1])
        atm = int(round(p10 / 50.0) * 50)

        signal = engine.generate_signal(slice_df, deadband=0.0015)
        bias = signal.get("final_bias", "SIDEWAYS_NO_TRADE")

        # Current week: DTE ~2.5 days on average
        # Next week: DTE ~9.5 days on average
        dte_cur_0 = 2.5
        dte_cur_1 = dte_cur_0 - 0.25 # decay in 90 mins
        dte_nxt_0 = 9.5
        dte_nxt_1 = dte_nxt_0 - 0.25

        if bias == "BEARISH":
            stk = atm + 200
            cur_pnl = bs_price(p10, stk, dte_cur_0, r, sigma, 'CE') - bs_price(p1130, stk, dte_cur_1, r, sigma, 'CE')
            nxt_pnl = bs_price(p10, stk, dte_nxt_0, r, sigma, 'CE') - bs_price(p1130, stk, dte_nxt_1, r, sigma, 'CE')
        elif bias == "BULLISH":
            stk = atm - 200
            cur_pnl = bs_price(p10, stk, dte_cur_0, r, sigma, 'PE') - bs_price(p1130, stk, dte_cur_1, r, sigma, 'PE')
            nxt_pnl = bs_price(p10, stk, dte_nxt_0, r, sigma, 'PE') - bs_price(p1130, stk, dte_nxt_1, r, sigma, 'PE')
        else: # SIDEWAYS
            stk_ce = atm + 200
            stk_pe = atm - 200
            cur_pnl = (bs_price(p10, stk_ce, dte_cur_0, r, sigma, 'CE') + bs_price(p10, stk_pe, dte_cur_0, r, sigma, 'PE')) - \
                      (bs_price(p1130, stk_ce, dte_cur_1, r, sigma, 'CE') + bs_price(p1130, stk_pe, dte_cur_1, r, sigma, 'PE'))
            nxt_pnl = (bs_price(p10, stk_ce, dte_nxt_0, r, sigma, 'CE') + bs_price(p10, stk_pe, dte_nxt_0, r, sigma, 'PE')) - \
                      (bs_price(p1130, stk_ce, dte_nxt_1, r, sigma, 'CE') + bs_price(p1130, stk_pe, dte_nxt_1, r, sigma, 'PE'))

        total += 1
        cur_pts_tot += cur_pnl
        next_pts_tot += nxt_pnl
        if cur_pnl > 0: cur_wins += 1
        if nxt_pnl > 0: next_wins += 1

        winner = "CURRENT WEEK ⚡" if cur_pnl > nxt_pnl else "NEXT WEEK 🛡️"
        print(f"{str(day):<11} | {p10:<10.1f} | {bias:<9} | {p1130:<10.1f} | {cur_pnl:>+8.2f} pts       | {nxt_pnl:>+8.2f} pts      | {winner}")

    print("-" * 130)
    print(f"\n🏆 HEAD-TO-HEAD COMPARISON (38 Sessions):")
    print(f"  • Current-Week Expiry : {cur_wins}/{total} Wins ({cur_wins/total*100:.1f}%) | Net Points: {cur_pts_tot:+.2f} pts (₹{cur_pts_tot*65:+,.2f})")
    print(f"  • Next-Week Expiry    : {next_wins}/{total} Wins ({next_wins/total*100:.1f}%) | Net Points: {next_pts_tot:+.2f} pts (₹{next_pts_tot*65:+,.2f})")
    print("=" * 130)

if __name__ == "__main__":
    run_comparison_current_vs_next()
