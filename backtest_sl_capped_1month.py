"""
Backtest: NIFTY 50 Intraday (10:00 AM -> 3:00 PM) with 50% SL & 80% Target Capping
===================================================================================
1-Month historical evaluation using real 5-minute candles.
Compares:
  A) NO SL (Hold until 3:00 PM blindly)
  B) WITH 50% STOP-LOSS & 80% PROFIT TARGET
"""

import sys
from pathlib import Path
from datetime import datetime, time as dtime, date, timedelta
import math
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from nifty_direction_engine.hybrid_engine import HybridSignalEngine

STEP = 50
LOT_SIZE = 65
RISK_FREE = 0.065
IV_BASE = 0.14
CAPITAL = 5_00_000
MARGIN_PER_LOT = 135_000
MAX_LOTS = int(CAPITAL // MARGIN_PER_LOT)  # 3 lots on 5 lakhs safe margin


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if opt == "CE" else max(K - S, 0)
        return max(intrinsic, 0.05)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def next_tuesday(d: date) -> date:
    days_ahead = (1 - d.weekday()) % 7
    return d + timedelta(days=days_ahead)


def run_sl_capped_backtest():
    print("=" * 115)
    print("  🎯  1-MONTH NIFTY 50 BACKTEST: 50% STOP-LOSS & 80% PROFIT TARGET CAPPING")
    print(f"  Capital: ₹{CAPITAL:,.0f} | Position: {MAX_LOTS} Lots (Naked Margin) | Holding: 10:00 AM -> 3:00 PM")
    print("=" * 115 + "\n")

    ticker = yf.Ticker("^NSEI")
    raw_df = ticker.history(period="1mo", interval="5m").reset_index()
    if raw_df.empty:
        print("[ERR] Could not load data.")
        return

    raw_df["timestamps"] = pd.to_datetime(raw_df["Datetime"]).dt.tz_localize(None)
    raw_df = raw_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    unique_days = sorted(set(raw_df["timestamps"].dt.date))

    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)

    print(f"{'DATE':<11} | {'10AM SPOT':<10} | {'BIAS':<9} | {'STRIKE':<15} | {'ENTRY':<7} | {'EXIT TYPE':<15} | {'EXIT TIME':<10} | {'NET PTS':<8} | {'WITH SL (₹)':<11} | {'NO SL (₹)'}")
    print("-" * 125)

    records = []
    lookback = 100

    for idx in range(2, len(unique_days)):
        trade_date = unique_days[idx]
        day_mask = raw_df["timestamps"].dt.date == trade_date
        day_candles = raw_df[day_mask]

        morn = day_candles[day_candles["timestamps"].dt.time <= dtime(10, 0)]
        rest = day_candles[(day_candles["timestamps"].dt.time > dtime(10, 0)) & (day_candles["timestamps"].dt.time <= dtime(15, 0))]

        if len(morn) < 5 or len(rest) < 10:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        x_df = raw_df.loc[cutoff_idx - lookback + 1: cutoff_idx].reset_index(drop=True)
        spot_10am = float(x_df["close"].iloc[-1])
        atm_strike = int(round(spot_10am / STEP) * STEP)

        # Signal
        try:
            signal_res = engine.generate_signal(x_df, deadband=0.0015)
        except Exception:
            continue

        bias = signal_res.get("final_bias", "SIDEWAYS_NO_TRADE")

        # Pick 4-OTM strikes (200 pts)
        if bias == "BULLISH":
            legs = [{"strike": atm_strike - 200, "type": "PE", "name": f"{atm_strike - 200} PE"}]
        elif bias == "BEARISH":
            legs = [{"strike": atm_strike + 200, "type": "CE", "name": f"{atm_strike + 200} CE"}]
        else:
            bias = "SIDEWAYS"
            legs = [
                {"strike": atm_strike + 200, "type": "CE", "name": f"{atm_strike + 200} CE"},
                {"strike": atm_strike - 200, "type": "PE", "name": f"{atm_strike - 200} PE"}
            ]

        expiry = next_tuesday(trade_date)
        cal_days = max((expiry - trade_date).days, 1)
        T_entry = cal_days / 365.0

        day_pts_with_sl = 0.0
        day_pts_no_sl = 0.0
        leg_statuses = []

        total_rest_bars = len(rest)

        for leg in legs:
            strike = leg["strike"]
            opt_type = leg["type"]

            entry_p = bs_price(spot_10am, strike, T_entry, RISK_FREE, IV_BASE, opt_type)
            sl_price = entry_p * 1.50   # 50% expansion
            tgt_price = entry_p * 0.20  # 80% decay

            leg_exit_p = entry_p
            exit_type = "3:00 PM EOD"
            exit_time = "15:00"

            # Check candle by candle up to 3:00 PM
            for bar_i, (_, candle) in enumerate(rest.iterrows()):
                c_close = float(candle["close"])
                c_time = candle["timestamps"].strftime("%H:%M")
                frac_day = bar_i / max(total_rest_bars, 1)
                T_curr = max(T_entry - (frac_day * (5.0 / 365.0)), 0.5 / 365.0)

                curr_p = bs_price(c_close, strike, T_curr, RISK_FREE, IV_BASE, opt_type)

                # Check SL
                if curr_p >= sl_price:
                    leg_exit_p = sl_price
                    exit_type = "STOP-LOSS (-50%)"
                    exit_time = c_time
                    break

                # Check TP
                if curr_p <= tgt_price:
                    leg_exit_p = tgt_price
                    exit_type = "TARGET (+80%)"
                    exit_time = c_time
                    break

                leg_exit_p = curr_p

            # PnL with disciplined SL/TP
            pts_with_sl = entry_p - leg_exit_p
            day_pts_with_sl += pts_with_sl

            # PnL with NO SL (held blindly to 3:00 PM)
            spot_3pm = float(rest["close"].iloc[-1])
            T_3pm = max(T_entry - (5.0 / 365.0), 0.5 / 365.0)
            exit_p_blind = bs_price(spot_3pm, strike, T_3pm, RISK_FREE, IV_BASE, opt_type)
            pts_no_sl = entry_p - exit_p_blind
            day_pts_no_sl += pts_no_sl

            leg_statuses.append((exit_type, exit_time, entry_p, leg["name"]))

        pnl_with_sl = day_pts_with_sl * LOT_SIZE * MAX_LOTS
        pnl_no_sl = day_pts_no_sl * LOT_SIZE * MAX_LOTS

        primary_status = leg_statuses[0][0]
        primary_time = leg_statuses[0][1]
        primary_entry = leg_statuses[0][2]
        strat_str = "/".join(l["name"] for l in legs)

        # If any leg hit SL, display it
        for st in leg_statuses:
            if "STOP-LOSS" in st[0]:
                primary_status = st[0]
                primary_time = st[1]
                break

        print(
            f"{str(trade_date):<11} | {spot_10am:<10.1f} | {bias:<9} | {strat_str:<15} | "
            f"{primary_entry:<7.1f} | {primary_status:<15} | {primary_time:<10} | "
            f"{day_pts_with_sl:>+6.1f}   | ₹{pnl_with_sl:>+9,.0f} | ₹{pnl_no_sl:>+9,.0f}"
        )

        records.append({
            "date": trade_date,
            "bias": bias,
            "pts_with_sl": day_pts_with_sl,
            "pts_no_sl": day_pts_no_sl,
            "pnl_with_sl": pnl_with_sl,
            "pnl_no_sl": pnl_no_sl,
            "exit_type": primary_status
        })

    # Summary
    df = pd.DataFrame(records)
    total = len(df)
    wins_sl = (df["pnl_with_sl"] > 0).sum()
    wins_no_sl = (df["pnl_no_sl"] > 0).sum()

    tot_pnl_sl = df["pnl_with_sl"].sum()
    tot_pnl_no_sl = df["pnl_no_sl"].sum()

    sl_triggers = sum(1 for r in records if "STOP-LOSS" in r["exit_type"])
    tp_triggers = sum(1 for r in records if "TARGET" in r["exit_type"])

    print("-" * 125)
    print("\n" + "=" * 65)
    print("  🏆  1-MONTH SCORECARD: 50% SL CAPPING vs BLIND HOLDING")
    print("=" * 65)
    print(f"  • Capital Pool               : ₹{CAPITAL:,.0f} ({MAX_LOTS} Lots Traded)")
    print(f"  • Total Evaluated Days       : {total} trading sessions")
    print(f"  • 🛑 Stop-Loss Triggers (-50%): {sl_triggers} sessions ({sl_triggers/total*100:.1f}%)")
    print(f"  • 🎯 Target Triggers (+80%)  : {tp_triggers} sessions ({tp_triggers/total*100:.1f}%)")
    print(f"  • ⏱️ Normal 3 PM Exits       : {total - sl_triggers - tp_triggers} sessions")
    print("  " + "─" * 60)
    print(f"  {'METRIC':<30} {'WITH 50% SL / 80% TP':>16} {'NO SL (BLIND)':>16}")
    print(f"  {'-'*62}")
    print(f"  {'Win Rate':<30} {wins_sl/total*100:>15.1f}% {wins_no_sl/total*100:>15.1f}%")
    print(f"  {'Net 1-Month PnL':<30} ₹{tot_pnl_sl:>14,.0f} ₹{tot_pnl_no_sl:>14,.0f}")
    print(f"  {'1-Month ROI':<30} {tot_pnl_sl/CAPITAL*100:>15.1f}% {tot_pnl_no_sl/CAPITAL*100:>15.1f}%")
    print(f"  {'Worst Single Day Loss':<30} ₹{df['pnl_with_sl'].min():>14,.0f} ₹{df['pnl_no_sl'].min():>14,.0f}")
    print(f"  {'Best Single Day Gain':<30} ₹{df['pnl_with_sl'].max():>14,.0f} ₹{df['pnl_no_sl'].max():>14,.0f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_sl_capped_backtest()
