"""
Backtest: Hedged Spread Strategy vs Naked Strategy — 1 Month (10:00 AM → 3:00 PM)
===================================================================================
Uses the same determine_intraday_option_strategy() now wired into daily_engine.py.
Option premiums are simulated via Black-Scholes (no broker required).
Compares:
  A) OLD  — Naked strangle / single naked leg
  B) NEW  — Hedged spreads (Bull Put / Bear Call / Iron Condor)
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

from options_engine import determine_intraday_option_strategy
from nifty_direction_engine.hybrid_engine import HybridSignalEngine

# ── Constants ──────────────────────────────────────────────────────────────────
STEP          = 50
LOT_SIZE      = 65          # Nifty lot size
RISK_FREE     = 0.065       # 6.5% risk-free rate
IV_BASE       = 0.14        # ~14% base IV for weekly Nifty options (realistic)

# Realistic Indian SEBI / NSE SPAN + Exposure margin requirements (Zerodha):
# - Naked short strangle (CE+PE): ~₹1,35,000 per lot (higher margin side + exposure)
# - Hedged Iron Condor (4 legs, 150pt wing): ~₹52,000 per lot (after ~60% hedge benefit)
# - Hedged Credit Spread (2 legs, 150pt wing): ~₹45,000 per lot
NAKED_MARGIN_PER_LOT  = 135_000
SPREAD_MARGIN_PER_LOT = 52_000
CAPITAL       = 5_00_000    # ₹5 lakh capital

# ── Black-Scholes helpers ──────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    """European Black-Scholes option price. T in years."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if opt == "CE" else max(K - S, 0)
        return max(intrinsic, 0.05)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def iv_for_strike(S: float, K: float, base_iv: float) -> float:
    """Simple volatility smile: wings get ~15% more IV than base."""
    moneyness = abs(K - S) / S
    skew = 1.0 + moneyness * 1.5
    return min(base_iv * skew, 0.60)


def trading_days_to_expiry(trade_date: date, expiry: date) -> float:
    """Time to expiry in years using trading day count."""
    cal_days = (expiry - trade_date).days
    return max(cal_days / 365.0, 1 / 365.0)


def next_tuesday(d: date) -> date:
    """Return the nearest upcoming Tuesday (same day if today is Tuesday)."""
    days_ahead = (1 - d.weekday()) % 7
    return d + timedelta(days=days_ahead)


def compute_realized_iv(prices: pd.Series, window: int = 20) -> float:
    """Annualised realized vol from recent log-return std."""
    if len(prices) < window + 1:
        return IV_BASE
    rets = np.log(prices / prices.shift(1)).dropna().iloc[-window:]
    return float(rets.std() * math.sqrt(252))

# ── Core backtest ──────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 105)
    print("  HEDGED SPREAD vs NAKED — 1-MONTH NIFTY INTRADAY BACKTEST  (10:00 AM → 3:00 PM)")
    print("  Strategy: NEW = Iron Condor / Bull-Put / Bear-Call  |  OLD = Naked Strangle/Leg")
    print("=" * 105 + "\n")

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print("[1/3] Downloading 1-month 5-minute NIFTY 50 data …")
    ticker = yf.Ticker("^NSEI")
    raw_df = ticker.history(period="1mo", interval="5m").reset_index()
    if raw_df.empty or len(raw_df) < 200:
        print("[ERR] Insufficient data from Yahoo Finance.")
        return

    raw_df["timestamps"] = pd.to_datetime(raw_df["Datetime"]).dt.tz_localize(None)
    raw_df = raw_df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    })
    raw_df = raw_df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()

    unique_days = sorted(set(raw_df["timestamps"].dt.date))
    print(f"      {len(unique_days)} trading days  |  {len(raw_df)} candles\n")

    # ── 2. Engine ──────────────────────────────────────────────────────────────
    print("[2/3] Initialising HybridSignalEngine (Kronos + LSTM) …")
    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)
    print("      Engine ready.\n")

    # ── 3. Backtest loop ───────────────────────────────────────────────────────
    print("[3/3] Running backtest loop …\n")

    HDR = (f"{'DATE':<11} {'SPOT':>8} {'BIAS':<10} "
           f"{'STRATEGY':<22} "
           f"{'NEW PnL':>8} {'OLD PnL':>8} "
           f"{'NEW LOTS':>8} {'OLD LOTS':>8} "
           f"{'3PM SPOT':>9} {'RESULT'}")
    print(HDR)
    print("-" * 115)

    records = []
    lookback = 100
    SKIP_FIRST = 2   # need context candles

    for idx in range(SKIP_FIRST, len(unique_days)):
        trade_date = unique_days[idx]

        day_mask    = raw_df["timestamps"].dt.date == trade_date
        day_candles = raw_df[day_mask]

        morn  = day_candles[day_candles["timestamps"].dt.time <= dtime(10, 0)]
        after = day_candles[day_candles["timestamps"].dt.time >  dtime(10, 0)]
        eod   = day_candles[day_candles["timestamps"].dt.time <= dtime(15, 0)]

        if len(morn) < 5 or len(after) < 5:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        x_df = raw_df.loc[cutoff_idx - lookback + 1: cutoff_idx].reset_index(drop=True)
        x_df = x_df[["open", "high", "low", "close", "volume"]].copy()

        spot_10am     = float(x_df["close"].iloc[-1])
        day_low_so_far = float(x_df["low"].min())
        day_high_so_far = float(x_df["high"].max())

        # ── Signal ────────────────────────────────────────────────────────────
        try:
            signal_res = engine.generate_signal(x_df, deadband=0.0015)
        except Exception as e:
            continue

        bias        = signal_res.get("final_bias", "SIDEWAYS_NO_TRADE")
        pred_return = signal_res.get("pred_return", 0.0)
        proj_close  = spot_10am * (1 + pred_return)

        # ── NEW: Hedged legs via options_engine ───────────────────────────────
        strategy = determine_intraday_option_strategy(
            current_price=spot_10am,
            day_low=day_low_so_far,
            day_high=day_high_so_far,
            proj_close=proj_close,
            step=STEP,
            otm_steps=4
        )
        strategy_name = {
            "BULLISH": "Bull Put Spread",
            "BEARISH": "Bear Call Spread",
            "NEUTRAL": "Iron Condor",
        }.get(strategy["bias"], "Iron Condor")

        expiry = next_tuesday(trade_date)
        T_entry = trading_days_to_expiry(trade_date, expiry)
        T_exit  = max(T_entry - (5 / 365.0), 0.5 / 365.0)  # ~5hrs later

        # realized IV from recent history
        hist_prices = raw_df.loc[:cutoff_idx, "close"]
        rv = compute_realized_iv(hist_prices)
        iv_today = max(rv * 1.10, IV_BASE)   # add 10% risk premium

        # 3 PM actual spot
        eod_3pm = day_candles[day_candles["timestamps"].dt.time <= dtime(15, 0)]
        spot_3pm = float(eod_3pm["close"].iloc[-1]) if not eod_3pm.empty else spot_10am

        # day's actual range
        actual_low  = float(after["low"].min())
        actual_high = float(after["high"].max())

        # ── Compute PnL for each strategy ─────────────────────────────────────

        def leg_pnl(action: str, strike: int, opt_type: str) -> float:
            """Net points per lot for one leg (positive = profit for the desk)."""
            iv = iv_for_strike(spot_10am, strike, iv_today)
            entry = bs_price(spot_10am, strike, T_entry, RISK_FREE, iv, opt_type)
            exit_ = bs_price(spot_3pm,  strike, T_exit,  RISK_FREE, iv, opt_type)
            if action == "SELL":
                return entry - exit_   # profit when premium decays
            else:  # BUY hedge
                return exit_ - entry   # hedge offsets some of sell profit

        # NEW: sum all legs
        new_legs     = strategy["legs"]
        new_pts      = sum(leg_pnl(lg["action"], lg["strike"], lg["type"]) for lg in new_legs)
        new_pnl_lot  = new_pts * LOT_SIZE

        # Margins
        new_lots     = max(1, int(CAPITAL // SPREAD_MARGIN_PER_LOT))
        old_legs_defs = _old_naked_legs(bias, spot_10am, day_low_so_far, day_high_so_far)
        old_pts      = sum(leg_pnl(lg["action"], lg["strike"], lg["type"]) for lg in old_legs_defs)
        old_pnl_lot  = old_pts * LOT_SIZE
        old_lots     = max(1, int(CAPITAL // NAKED_MARGIN_PER_LOT))

        # Total PnL across all lots
        new_total = new_pnl_lot * new_lots
        old_total = old_pnl_lot * old_lots

        result_str = "✅ WIN" if new_pts > 0 else "❌ LOSS"

        print(
            f"{str(trade_date):<11} {spot_10am:>8.0f} {strategy['bias']:<10} "
            f"{strategy_name:<22} "
            f"{new_total:>+8.0f} {old_total:>+8.0f} "
            f"{new_lots:>8} {old_lots:>8} "
            f"{spot_3pm:>9.0f} {result_str}"
        )

        records.append({
            "date":         trade_date,
            "bias":         strategy["bias"],
            "strategy":     strategy_name,
            "spot_10am":    spot_10am,
            "spot_3pm":     spot_3pm,
            "new_pts":      new_pts,
            "old_pts":      old_pts,
            "new_lots":     new_lots,
            "old_lots":     old_lots,
            "new_total":    new_total,
            "old_total":    old_total,
            "win":          new_pts > 0,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if not records:
        print("[WARN] No sessions evaluated.")
        return

    df = pd.DataFrame(records)

    wins         = df["win"].sum()
    total        = len(df)
    new_cum      = df["new_total"].sum()
    old_cum      = df["old_total"].sum()
    new_avg_day  = df["new_total"].mean()
    old_avg_day  = df["old_total"].mean()
    best_day     = df.loc[df["new_total"].idxmax()]
    worst_day    = df.loc[df["new_total"].idxmin()]
    avg_new_lots = df["new_lots"].mean()
    avg_old_lots = df["old_lots"].mean()

    # Max drawdown (running cumulative)
    cumulative = df["new_total"].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_dd = drawdown.min()

    print("-" * 115)
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS — HEDGED vs NAKED  ({total} trading days)")
    print(f"{'='*60}")
    print(f"  Capital Deployed          : ₹{CAPITAL:,.0f}")
    print(f"  Lot Size                  : {LOT_SIZE} units")
    print()
    print(f"  {'METRIC':<35} {'HEDGED (NEW)':>14} {'NAKED (OLD)':>14}")
    print(f"  {'-'*63}")
    print(f"  {'Avg Lots per Day':<35} {avg_new_lots:>14.1f} {avg_old_lots:>14.1f}")
    print(f"  {'Win Rate':<35} {wins/total*100:>13.1f}%  {'—':>14}")
    print(f"  {'Avg Daily PnL (all lots)':<35} ₹{new_avg_day:>12,.0f}  ₹{old_avg_day:>12,.0f}")
    print(f"  {'Net Monthly PnL':<35} ₹{new_cum:>12,.0f}  ₹{old_cum:>12,.0f}")
    print(f"  {'Monthly ROI on Capital':<35} {new_cum/CAPITAL*100:>13.1f}%  {old_cum/CAPITAL*100:>13.1f}%")
    print(f"  {'Max Drawdown (single month)':<35} ₹{max_dd:>12,.0f}")
    print(f"  {'Margin per lot (SPAN est.)':<35} ₹{SPREAD_MARGIN_PER_LOT:>12,.0f}  ₹{NAKED_MARGIN_PER_LOT:>12,.0f}")
    print()
    print(f"  Best Day  : {best_day['date']}  [{best_day['strategy']}]  ₹{best_day['new_total']:+,.0f}")
    print(f"  Worst Day : {worst_day['date']}  [{worst_day['strategy']}]  ₹{worst_day['new_total']:+,.0f}")
    print(f"\n  Lot leverage gain (Hedged/Naked): {avg_new_lots/avg_old_lots:.1f}x more lots on same capital")
    print(f"{'='*60}\n")

    # Strategy breakdown
    print("  BREAKDOWN BY STRATEGY:")
    print(f"  {'Strategy':<22} {'Days':>5} {'Win%':>7} {'Avg PnL/Day':>14} {'Total PnL':>12}")
    print(f"  {'-'*62}")
    for strat, grp in df.groupby("strategy"):
        w  = grp["win"].sum()
        n  = len(grp)
        wp = w / n * 100
        ap = grp["new_total"].mean()
        tp = grp["new_total"].sum()
        print(f"  {strat:<22} {n:>5} {wp:>6.1f}% ₹{ap:>12,.0f}  ₹{tp:>10,.0f}")
    print()


def _old_naked_legs(bias: str, spot: float, day_low: float, day_high: float) -> list:
    """Replicate the old daily_engine naked leg selection for comparison."""
    atm = int(round(spot / STEP) * STEP)
    otm = 200  # 4 * 50

    if bias == "BULLISH":
        return [{"action": "SELL", "strike": atm - otm, "type": "PE"}]
    elif bias == "BEARISH":
        return [{"action": "SELL", "strike": atm + otm, "type": "CE"}]
    else:  # SIDEWAYS / NO_TRADE
        return [
            {"action": "SELL", "strike": atm + otm, "type": "CE"},
            {"action": "SELL", "strike": atm - otm, "type": "PE"},
        ]


if __name__ == "__main__":
    run_backtest()
