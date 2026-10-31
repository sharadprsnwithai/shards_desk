"""
2-Month Quantitative Backtest: NIFTY 50 Intraday Strategy (Updated Rules)
========================================================================
Parameters:
  - Index: NIFTY 50 (^NSEI), 5-minute candles
  - Lot Size: 65
  - Expiry: Nearest Tuesday (holiday-aware shift to Monday)
  - Entry: 10:00 AM IST
  - Signal: Hybrid Engine (Kronos-base + LSTM + VWAP Acceptance + Deadband 0.15%)
  - Structure: 4-OTM (200-point buffer) + 150-pt Hedge Wing
      * BULLISH  -> Bull Put Credit Spread (Sell -200 PE, Buy -350 PE)
      * BEARISH  -> Bear Call Credit Spread (Sell +200 CE, Buy +350 CE)
      * SIDEWAYS -> Iron Condor (Sell -200 PE / Buy -350 PE & Sell +200 CE / Buy +350 CE)
  - Management:
      * 50% Stop-Loss on sold leg (accompanied by wing square-off)
      * 80% Profit Target on sold leg
      * 3:00 PM EOD Hard Exit on any remaining open legs
  - Event Filter: Skip scheduled high-impact events
  - Capital: ₹5,00,000 margin pool (Margin ~₹55,000 per lot)
  - Costs: Brokerage + STT + Turnover + GST + Slippage (~₹100 per lot round-trip)
"""

import sys
import math

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from pathlib import Path
from datetime import datetime, time as dtime, date, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

from nifty_direction_engine.hybrid_engine import HybridSignalEngine
from event_filter import check_event_risk, should_skip_trading
from market_calendar import is_nse_trading_day

# ── Parameters ───────────────────────────────────────────────────────────────
STEP = 50
OTM_DISTANCE = 200
WING_DISTANCE = 150
LOT_SIZE = 65
RISK_FREE = 0.065
IV_BASE = 0.14
CAPITAL = 500_000.0
MARGIN_PER_SPREAD = 55_000.0
MAX_LOTS = int(CAPITAL // MARGIN_PER_SPREAD)  # ~9 lots on ₹5L capital

COST_PER_SPREAD_LOT = 85.0  # ₹85 round-trip friction (brokerage ₹40 + STT/taxes + slippage)

# ── Black-Scholes Pricing Helpers ──────────────────────────────────────────

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
    """Volatility smile: OTM wings trade at slight premium skew."""
    moneyness = abs(K - S) / S
    skew = 1.0 + moneyness * 1.5
    return min(base_iv * skew, 0.55)


def get_target_tuesday_expiry(trade_date: date) -> date:
    """Closest upcoming Tuesday expiry, shifted back if holiday."""
    days_ahead = (1 - trade_date.weekday()) % 7  # 1 = Tuesday
    target_d = trade_date + timedelta(days=days_ahead)
    while not is_nse_trading_day(target_d) and target_d > trade_date:
        target_d -= timedelta(days=1)
    return target_d


def compute_realized_iv(prices: pd.Series, window: int = 20) -> float:
    if len(prices) < window + 1:
        return IV_BASE
    rets = np.log(prices / prices.shift(1)).dropna().iloc[-window:]
    return float(max(0.10, min(rets.std() * math.sqrt(252), 0.35)))

# ── Main Backtest Runner ───────────────────────────────────────────────────

def run_2month_backtest():
    print("=" * 115)
    print("  🚀  NIFTY 50 INTRADAY OPTION DESK — 2-MONTH QUANTITATIVE BACKTEST")
    print(f"  Configuration: Lot Size = {LOT_SIZE} | Tuesday Expiry | 4-OTM (200-pt) + 150-pt Wing")
    print("  Rules: 50% SL / 80% TP / 3:00 PM EOD Square-Off | Defined-Risk Credit Spreads")
    print("=" * 115 + "\n")

    # 1. Download 2-month 5m candles (60 days)
    print("[1/3] Downloading 60-day 5m candles for NIFTY 50 (^NSEI) from Yahoo Finance...")
    ticker = yf.Ticker("^NSEI")
    raw_df = ticker.history(period="60d", interval="5m").reset_index()

    if raw_df.empty or len(raw_df) < 500:
        print("❌ Error: Insufficient market data fetched.")
        return

    raw_df["timestamps"] = pd.to_datetime(raw_df["Datetime"]).dt.tz_localize(None)
    raw_df = raw_df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    })
    raw_df = raw_df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()

    unique_days = sorted(list(set(raw_df["timestamps"].dt.date)))
    print(f"      Loaded {len(unique_days)} total trading sessions from {unique_days[0]} to {unique_days[-1]}.\n")

    # 2. Initialize Signal Engine
    print("[2/3] Initializing HybridSignalEngine (Kronos Transformer + Microstructure LSTM)...")
    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)
    print("      Engine calibrated.\n")

    # 3. Simulate Day-by-Day Path
    print("[3/3] Simulating intraday execution candle-by-candle (10:00 AM → 3:00 PM)...\n")

    header = (
        f"{'DATE':<11} | {'DAY':<3} | {'10AM SPOT':<9} | {'BIAS':<9} | {'STRATEGY':<18} | "
        f"{'NET PTS':<8} | {'EXIT TRIGGER':<17} | {'EXIT TIME':<9} | {'1-LOT PnL':<11} | {'PORTFOLIO PnL'}"
    )
    print(header)
    print("-" * 115)

    records = []
    lookback = 100

    for idx in range(2, len(unique_days)):
        trade_date = unique_days[idx]
        weekday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][trade_date.weekday()]

        # Event Circuit Breaker
        if should_skip_trading(trade_date):
            print(f"{str(trade_date):<11} | {weekday_name:<3} | --- EVENT DAY SKIPPED (Macro Circuit Breaker) ---")
            continue

        day_mask = raw_df["timestamps"].dt.date == trade_date
        day_candles = raw_df[day_mask]

        morn = day_candles[day_candles["timestamps"].dt.time <= dtime(10, 0)]
        after = day_candles[(day_candles["timestamps"].dt.time > dtime(10, 0)) & (day_candles["timestamps"].dt.time <= dtime(15, 0))]

        if len(morn) < 5 or len(after) < 10:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        x_df = raw_df.loc[cutoff_idx - lookback + 1: cutoff_idx].reset_index(drop=True)
        spot_10am = float(x_df["close"].iloc[-1])
        atm_strike = int(round(spot_10am / STEP) * STEP)

        # ── Signal Generation at 10:00 AM ──
        try:
            signal_res = engine.generate_signal(x_df, deadband=0.0015)
            bias = signal_res.get("final_bias", "SIDEWAYS_NO_TRADE")
        except Exception as e:
            bias = "SIDEWAYS"

        bias = bias if bias in ("BULLISH", "BEARISH") else "SIDEWAYS"

        # ── Build Spreads according to Strategy ──
        # Spreads: (sold_strike, sold_type, wing_strike, wing_type)
        if bias == "BULLISH":
            strategy_name = "Bull Put Spread"
            spread_pairs = [
                (atm_strike - OTM_DISTANCE, "PE", atm_strike - OTM_DISTANCE - WING_DISTANCE, "PE")
            ]
        elif bias == "BEARISH":
            strategy_name = "Bear Call Spread"
            spread_pairs = [
                (atm_strike + OTM_DISTANCE, "CE", atm_strike + OTM_DISTANCE + WING_DISTANCE, "CE")
            ]
        else:
            strategy_name = "Iron Condor"
            spread_pairs = [
                (atm_strike - OTM_DISTANCE, "PE", atm_strike - OTM_DISTANCE - WING_DISTANCE, "PE"),
                (atm_strike + OTM_DISTANCE, "CE", atm_strike + OTM_DISTANCE + WING_DISTANCE, "CE")
            ]

        # Expiry setup
        expiry = get_target_tuesday_expiry(trade_date)
        cal_days = max((expiry - trade_date).days, 0)
        # 0-DTE handling for Tuesday
        T_entry = max(cal_days / 365.0, 0.5 / 365.0)

        # Volatility
        hist_prices = raw_df.loc[:cutoff_idx, "close"]
        rv = compute_realized_iv(hist_prices)
        iv_today = max(rv * 1.05, IV_BASE)

        # ── Path Simulation for each spread ──
        total_session_candles = len(after)
        session_net_pts = 0.0
        session_triggers = []

        for sold_strike, sold_type, wing_strike, wing_type in spread_pairs:
            sold_iv = iv_for_strike(spot_10am, sold_strike, iv_today)
            wing_iv = iv_for_strike(spot_10am, wing_strike, iv_today)

            sold_entry_p = bs_price(spot_10am, sold_strike, T_entry, RISK_FREE, sold_iv, sold_type)
            wing_entry_p = bs_price(spot_10am, wing_strike, T_entry, RISK_FREE, wing_iv, wing_type)

            net_credit_entry = sold_entry_p - wing_entry_p

            sl_price = sold_entry_p * 1.50   # 50% expansion of sold premium
            tgt_price = sold_entry_p * 0.20  # 80% decay of sold premium

            exit_trigger = "3:00 PM EOD"
            exit_time = "15:00"
            sold_exit_p = sold_entry_p
            wing_exit_p = wing_entry_p

            # 5-minute candle by candle simulation
            for c_idx, (_, candle) in enumerate(after.iterrows()):
                candle_close = float(candle["close"])
                candle_time = candle["timestamps"].strftime("%H:%M")

                frac_day = c_idx / max(total_session_candles, 1)
                T_curr = max(T_entry - (frac_day * (5.0 / 365.0)), 0.1 / 365.0)

                curr_sold_p = bs_price(candle_close, sold_strike, T_curr, RISK_FREE, sold_iv, sold_type)
                curr_wing_p = bs_price(candle_close, wing_strike, T_curr, RISK_FREE, wing_iv, wing_type)

                # Check Stop-Loss (+50% expansion)
                if curr_sold_p >= sl_price:
                    exit_trigger = "STOP-LOSS (-50%)"
                    exit_time = candle_time
                    sold_exit_p = sl_price
                    wing_exit_p = curr_wing_p
                    break

                # Check Profit Target (+80% decay)
                if curr_sold_p <= tgt_price:
                    exit_trigger = "TARGET (+80%)"
                    exit_time = candle_time
                    sold_exit_p = tgt_price
                    wing_exit_p = curr_wing_p
                    break

                sold_exit_p = curr_sold_p
                wing_exit_p = curr_wing_p

            # Spread PnL: (Sold_Entry - Sold_Exit) + (Wing_Exit - Wing_Entry)
            sold_leg_pts = sold_entry_p - sold_exit_p
            wing_leg_pts = wing_exit_p - wing_entry_p
            spread_pts = sold_leg_pts + wing_leg_pts

            session_net_pts += spread_pts
            session_triggers.append((exit_trigger, exit_time))

        # Overall session trigger
        primary_trigger = session_triggers[0][0]
        primary_time = session_triggers[0][1]
        for tr, tm in session_triggers:
            if "STOP-LOSS" in tr:
                primary_trigger = tr
                primary_time = tm
                break

        # Sizing and PnL
        num_spreads = len(spread_pairs)
        single_lot_gross = session_net_pts * LOT_SIZE
        single_lot_net = single_lot_gross - (COST_PER_SPREAD_LOT * num_spreads)

        # Portfolio sizing (allocates available capital across lots)
        lots_traded = max(1, int(CAPITAL // (MARGIN_PER_SPREAD * num_spreads)))
        portfolio_gross = session_net_pts * LOT_SIZE * lots_traded
        portfolio_net = portfolio_gross - (COST_PER_SPREAD_LOT * num_spreads * lots_traded)

        print(
            f"{str(trade_date):<11} | {weekday_name:<3} | {spot_10am:<9.1f} | {bias:<9} | {strategy_name:<18} | "
            f"{session_net_pts:>+6.2f} pts | {primary_trigger:<17} | {primary_time:<9} | "
            f"₹{single_lot_net:>+8,.0f}   | ₹{portfolio_net:>+10,.0f} ({lots_traded}L)"
        )

        records.append({
            "date": trade_date,
            "day": weekday_name,
            "spot_10am": spot_10am,
            "bias": bias,
            "strategy": strategy_name,
            "net_pts": session_net_pts,
            "trigger": primary_trigger,
            "exit_time": primary_time,
            "single_lot_net": single_lot_net,
            "portfolio_net": portfolio_net,
            "lots": lots_traded,
            "win": session_net_pts > 0
        })

    # ── Analytics & Institutional Performance Scorecard ──
    print("-" * 115)
    if not records:
        print("❌ No records generated.")
        return

    df_res = pd.DataFrame(records)

    total_sessions = len(df_res)
    win_sessions = df_res["win"].sum()
    loss_sessions = total_sessions - win_sessions
    win_rate = (win_sessions / total_sessions) * 100.0

    target_count = sum(1 for r in records if "TARGET" in r["trigger"])
    sl_count = sum(1 for r in records if "STOP-LOSS" in r["trigger"])
    eod_count = sum(1 for r in records if "EOD" in r["trigger"])

    total_pts = df_res["net_pts"].sum()
    total_single_lot_net = df_res["single_lot_net"].sum()
    total_portfolio_net = df_res["portfolio_net"].sum()

    gains = df_res[df_res["single_lot_net"] > 0]["single_lot_net"].sum()
    losses = abs(df_res[df_res["single_lot_net"] < 0]["single_lot_net"].sum())
    profit_factor = (gains / losses) if losses > 0 else float("inf")

    avg_win_pts = df_res[df_res["net_pts"] > 0]["net_pts"].mean() if win_sessions > 0 else 0.0
    avg_loss_pts = abs(df_res[df_res["net_pts"] < 0]["net_pts"].mean()) if loss_sessions > 0 else 0.0

    # Drawdown
    equity_curve = df_res["portfolio_net"].cumsum() + CAPITAL
    peak = equity_curve.cummax()
    drawdown = equity_curve - peak
    max_dd_rs = drawdown.min()
    max_dd_pct = (max_dd_rs / CAPITAL) * 100.0

    total_roi = (total_portfolio_net / CAPITAL) * 100.0

    print("\n" + "=" * 65)
    print("  🏆  NIFTY 50 2-MONTH INTRADAY SCORECARD (UPDATED RULES)")
    print("=" * 65)
    print(f"  • Backtest Window               : {total_sessions} Trading Days")
    print(f"  • Base Margin Capital Pool       : ₹{CAPITAL:,.0f}")
    print(f"  • Standard Contract Lot Size     : {LOT_SIZE} units")
    print(f"  • Option Expiry Calibration      : Nearest Tuesday (Holiday Shifted)")
    print("-" * 65)
    print(f"  • Winning Sessions               : {win_sessions} / {total_sessions} ({win_rate:.1f}%)")
    print(f"  • Target Booked (+80% Decay)     : {target_count} sessions ({(target_count / total_sessions) * 100:.1f}%)")
    print(f"  • Stop-Loss Triggered (-50% Cut) : {sl_count} sessions ({(sl_count / total_sessions) * 100:.1f}%)")
    print(f"  • Normal 3:00 PM EOD Exits       : {eod_count} sessions ({(eod_count / total_sessions) * 100:.1f}%)")
    print("-" * 65)
    print(f"  • Net Points Captured            : {total_pts:+.2f} index points")
    print(f"  • Net 1-Lot P&L (after all fees) : ₹{total_single_lot_net:+,.2f}")
    print(f"  • Net Portfolio P&L (₹5L Margin) : ₹{total_portfolio_net:+,.2f}")
    print(f"  • Total 2-Month ROI on Margin    : {total_roi:+.2f}% (~{total_roi / 2:+.2f}% / month)")
    print(f"  • Profit Factor (Gains / Losses) : {profit_factor:.2f}")
    print(f"  • Realized Win / Loss Ratio      : {avg_win_pts:.2f} pts : {avg_loss_pts:.2f} pts")
    print(f"  • Peak Portfolio Drawdown        : ₹{abs(max_dd_rs):,.2f} ({abs(max_dd_pct):.2f}%)")
    print("=" * 65 + "\n")

    # Strategy breakdown
    print("📊 Performance Breakdown by Sub-Strategy:")
    for strat in ["Bull Put Spread", "Bear Call Spread", "Iron Condor"]:
        sub = df_res[df_res["strategy"] == strat]
        if not sub.empty:
            sub_wins = sub["win"].sum()
            sub_rate = (sub_wins / len(sub)) * 100.0
            sub_pts = sub["net_pts"].sum()
            sub_pnl = sub["portfolio_net"].sum()
            print(f"  • {strat:<18} : {len(sub):>2} trades | Win: {sub_rate:5.1f}% | Pts: {sub_pts:>+6.1f} | Net ₹: {sub_pnl:>+9,.0f}")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    run_2month_backtest()
