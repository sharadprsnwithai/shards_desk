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


def run_backtest_with_50pct_sl():
    print("=" * 135)
    print("  📊  NIFTY 50 1-MONTH TEST: 3:00 PM HOLD WITH AUTOMATIC 50% GTT STOP-LOSS")
    print("  Strategy: 4-OTM (200-Point Buffer) Entered at 10:00 AM IST | Current-Week Expiry")
    print("  Rule: If Option Premium expands by >= 50% above entry at ANY 5-minute bar -> STOP-LOSS TRIGGERED")
    print("  Otherwise: Hold until 3:00 PM IST to capture full daily theta decay")
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

    tot_pts = 0.0
    wins = 0
    losses = 0
    sl_triggered_count = 0
    full_decay_count = 0
    total_sessions = 0

    print(f"{'DATE':<11} | {'10AM SPOT':<10} | {'BIAS':<9} | {'10:00 PREM':<12} | {'EXIT PREM':<12} | {'PNL (PTS)':<10} | {'EXIT TIME':<10} | {'EXIT REASON':<22} | {'VERDICT'}")
    print("-" * 135)

    for day in unique_days:
        day_df = df[df["timestamps"].dt.date == day]
        morn = day_df[day_df["timestamps"].dt.time <= dtime(10, 0)]
        holding_bars = day_df[(day_df["timestamps"].dt.time > dtime(10, 0)) & (day_df["timestamps"].dt.time <= dtime(15, 0))]

        if len(morn) < 5 or len(holding_bars) < 20:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        slice_df = df.loc[cutoff_idx - lookback + 1:cutoff_idx].copy().reset_index(drop=True)
        p10 = float(morn["close"].iloc[-1])
        atm = int(round(p10 / 50.0) * 50)

        # Generate Signal at 10:00 AM
        signal = engine.generate_signal(slice_df, deadband=0.0015)
        bias = signal.get("final_bias", "SIDEWAYS_NO_TRADE")

        dte_1000 = 2.5

        # Determine strikes
        if bias == "BEARISH":
            ce_stk = atm + 200
            pe_stk = None
            p0 = bs_price(p10, ce_stk, dte_1000, r, sigma, 'CE')
        elif bias == "BULLISH":
            ce_stk = None
            pe_stk = atm - 200
            p0 = bs_price(p10, pe_stk, dte_1000, r, sigma, 'PE')
        else:  # SIDEWAYS
            bias = "SIDEWAYS"
            ce_stk = atm + 200
            pe_stk = atm - 200
            p0 = bs_price(p10, ce_stk, dte_1000, r, sigma, 'CE') + bs_price(p10, pe_stk, dte_1000, r, sigma, 'PE')

        # 50% Stop-Loss threshold (Option expands to 1.50x of entry premium)
        sl_threshold = p0 * 1.50
        max_loss_pts = -(p0 * 0.50)

        exit_prem = None
        exit_time_str = "15:00 PM"
        exit_reason = "Target 3:00 PM Decay"
        sl_hit = False

        # Step through every 5-minute candle between 10:05 and 15:00
        total_bars = len(holding_bars)
        for b_idx, (_, row) in enumerate(holding_bars.iterrows()):
            spot_bar = float(row["close"])
            t_bar = row["timestamps"].time()

            # Fraction of 5-hour day elapsed
            elapsed_days = (b_idx + 1) * (5.0 / 60.0) / 6.0
            dte_current = max(dte_1000 - elapsed_days, 1e-4)

            # Current option premium
            if bias == "BEARISH":
                prem_bar = bs_price(spot_bar, ce_stk, dte_current, r, sigma, 'CE')
            elif bias == "BULLISH":
                prem_bar = bs_price(spot_bar, pe_stk, dte_current, r, sigma, 'PE')
            else:
                prem_bar = bs_price(spot_bar, ce_stk, dte_current, r, sigma, 'CE') + \
                           bs_price(spot_bar, pe_stk, dte_current, r, sigma, 'PE')

            # Check if 50% Stop-Loss breached
            if prem_bar >= sl_threshold:
                sl_hit = True
                exit_prem = sl_threshold
                exit_time_str = t_bar.strftime("%H:%M")
                exit_reason = "50% GTT Stop-Loss Hit"
                break

        # If not stopped out, exit at 3:00 PM
        if not sl_hit:
            last_bar = holding_bars.iloc[-1]
            last_spot = float(last_bar["close"])
            dte_1500 = dte_1000 - (5.0 / 6.0)
            if bias == "BEARISH":
                exit_prem = bs_price(last_spot, ce_stk, dte_1500, r, sigma, 'CE')
            elif bias == "BULLISH":
                exit_prem = bs_price(last_spot, pe_stk, dte_1500, r, sigma, 'PE')
            else:
                exit_prem = bs_price(last_spot, ce_stk, dte_1500, r, sigma, 'CE') + \
                            bs_price(last_spot, pe_stk, dte_1500, r, sigma, 'PE')

        # Profit for option seller = Entry - Exit
        pts = p0 - exit_prem

        total_sessions += 1
        tot_pts += pts

        if pts > 0:
            wins += 1
            verdict = "WIN ✅"
            full_decay_count += 1
        else:
            losses += 1
            verdict = "LOSS 🛑" if sl_hit else "LOSS ❌"
            if sl_hit:
                sl_triggered_count += 1

        print(f"{str(day):<11} | {p10:<10.1f} | {bias:<9} | ₹{p0:>9.2f}  | ₹{exit_prem:>9.2f}  | {pts:>+7.2f} pt | {exit_time_str:<10} | {exit_reason:<22} | {verdict}")

    print("-" * 135)
    print("\n" + "=" * 65)
    print("      🏆 3:00 PM HOLD + 50% GTT STOP-LOSS AUDIT (38 SESSIONS)")
    print("=" * 65)
    print(f"  Total Trading Sessions   : {total_sessions}")
    print(f"  Profitable Sessions (Wins): {wins} ({wins/total_sessions*100:.1f}%)")
    print(f"  Losing Sessions          : {losses} ({losses/total_sessions*100:.1f}%)")
    print(f"  Stop-Losses Triggered    : {sl_triggered_count} (Controlled loss capped at -50%)")
    print(f"  Full Decay Captured (3PM): {full_decay_count}")
    print(f"  Net Total Points Gained  : {tot_pts:>+8.2f} pts")
    print(f"  Average Points Per Day   : {tot_pts/total_sessions:>+8.2f} pts / day")
    print(f"  Approx Net PnL (1 Lot)   : ₹{tot_pts * 65:>+8,.2f}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_backtest_with_50pct_sl()
