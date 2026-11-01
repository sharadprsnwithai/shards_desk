"""
backtest_jan2025_kite.py
========================
Runs two backtests for January 2025 using REAL Zerodha Kite API historical data:

  1. DAILY STRATEGY  — Nifty 50 daily direction prediction + option selling safety assessment
  2. MOMENTUM STRATEGY — Sector Momentum intraday strategy (Phase 1 + Phase 2)

Data source: Zerodha Kite Connect API (real market data, no synthetic data).
"""
from __future__ import annotations

import sys
import os
import logging
import time as time_module
from datetime import datetime, timedelta, date, time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
JAN_2025_START = datetime(2025, 1, 1)
JAN_2025_END   = datetime(2025, 1, 31)

# Kite instrument tokens (pre-mapped)
INDEX_TOKENS: dict[str, int] = {
    "NIFTY":   256265,
    "IT":      259849,
    "Bank":    260105,
    "Auto":    263433,
    "Pharma":  262409,
    "FMCG":    261897,
    "Metal":   263689,
    "Realty":  261129,
    "Energy":  261641,
}

SECTOR_STOCKS: dict[str, list[tuple[str, str, int]]] = {
    # sector -> list of (yf_ticker, nse_symbol, approx_token)
    "IT": [
        ("TCS.NS",     "TCS",     2953217),
        ("INFY.NS",    "INFY",    408065),
        ("WIPRO.NS",   "WIPRO",   969473),
        ("HCLTECH.NS", "HCLTECH", 1850625),
        ("TECHM.NS",   "TECHM",   3465729),
        ("COFORGE.NS", "COFORGE", 3350721),
    ],
    "Bank": [
        ("HDFCBANK.NS",  "HDFCBANK",  341249),
        ("ICICIBANK.NS", "ICICIBANK", 1270529),
        ("KOTAKBANK.NS", "KOTAKBANK", 492033),
        ("AXISBANK.NS",  "AXISBANK",  1510401),
        ("SBIN.NS",      "SBIN",      779521),
        ("BANKBARODA.NS","BANKBARODA",1195009),
    ],
    "Auto": [
        ("MARUTI.NS",    "MARUTI",    2815745),
        ("TVSMOTOR.NS",  "TVSMOTOR",  2170625),
        ("BAJAJ-AUTO.NS","BAJAJ-AUTO",4267265),
        ("HEROMOTOCO.NS","HEROMOTOCO",345089),
        ("EICHERMOT.NS", "EICHERMOT", 232961),
        ("M&M.NS",       "M&M",       519937),
    ],
    "Pharma": [
        ("SUNPHARMA.NS", "SUNPHARMA",  857857),
        ("DRREDDY.NS",   "DRREDDY",    225537),
        ("CIPLA.NS",     "CIPLA",      177665),
        ("DIVISLAB.NS",  "DIVISLAB",   2800641),
        ("AUROPHARMA.NS","AUROPHARMA", 70401),
        ("BIOCON.NS",    "BIOCON",     2911489),
    ],
    "FMCG": [
        ("HINDUNILVR.NS","HINDUNILVR",356865),
        ("ITC.NS",       "ITC",       424961),
        ("NESTLEIND.NS", "NESTLEIND", 4598529),
        ("BRITANNIA.NS", "BRITANNIA", 140033),
        ("DABUR.NS",     "DABUR",     197633),
        ("GODREJCP.NS",  "GODREJCP",  3463169),
    ],
    "Metal": [
        ("TATASTEEL.NS", "TATASTEEL", 895745),
        ("HINDALCO.NS",  "HINDALCO",  348929),
        ("JSWSTEEL.NS",  "JSWSTEEL",  3001089),
        ("VEDL.NS",      "VEDL",      784129),
        ("NATIONALUM.NS","NATIONALUM",4592641),
        ("NMDC.NS",      "NMDC",      3924993),
    ],
    "Realty": [
        ("DLF.NS",       "DLF",       3771393),
        ("LODHA.NS",     "LODHA",     5552641),
        ("OBEROIRLTY.NS","OBEROIRLTY",3876097),
        ("PRESTIGE.NS",  "PRESTIGE",  4278529),
        ("GODREJPROP.NS","GODREJPROP",3463425),
        ("SOBHA.NS",     "SOBHA",     857345),
    ],
    "Energy": [
        ("RELIANCE.NS",  "RELIANCE",  738561),
        ("ONGC.NS",      "ONGC",      633601),
        ("BPCL.NS",      "BPCL",      134657),
        ("IOC.NS",       "IOC",       415745),
        ("NTPC.NS",      "NTPC",      2977281),
        ("POWERGRID.NS", "POWERGRID", 3834113),
    ],
}

# Strategy config (mirrors sector_momentum_desk/config.py)
BIAS_THRESHOLD         = 0.0015
SL_BUFFER              = 0.0005
EMA_TRAIL_PERIOD       = 10
USE_EMA_TRAILING       = True
COST_BPS_ROUNDTRIP     = 25.0
BREAKOUT_SLIPPAGE_PCT  = 0.05
NUM_TOP_STOCKS         = 3
NIFTY_STEP             = 50   # option strike step for daily strategy

CACHE_DIR = Path(__file__).resolve().parent / "data_cache" / "jan2025_kite"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# KITE DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def get_kite():
    """Returns an authenticated KiteConnect client."""
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key=os.getenv("KITE_API_KEY", "").strip())
    kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN", "").strip())
    try:
        kite.profile()
        return kite
    except Exception:
        from kite_auth import auto_login_with_totp
        token = auto_login_with_totp()
        if not token:
            raise RuntimeError("Cannot authenticate with Kite. Check .env credentials.")
        kite.set_access_token(token)
        return kite


def fetch_kite_daily(kite, token: int, symbol: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Fetch daily OHLCV for a symbol (Kite token) and return a clean DataFrame."""
    cache_file = CACHE_DIR / f"{symbol}_{token}_daily.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        return df

    try:
        recs = kite.historical_data(token, from_dt - timedelta(days=5), to_dt, interval="day")
        time_module.sleep(0.3)
        if not recs:
            return pd.DataFrame()
        df = pd.DataFrame(recs)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.rename(columns={"date": "timestamps", "open": "Open", "high": "High",
                                 "low": "Low", "close": "Close", "volume": "Volume"})
        df = df.sort_values("timestamps").reset_index(drop=True)
        df.to_parquet(cache_file)
        return df
    except Exception as e:
        logger.warning(f"Daily fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_kite_5m(kite, token: int, symbol: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Fetch 5-minute OHLCV in 60-day chunks and return a clean DataFrame."""
    cache_file = CACHE_DIR / f"{symbol}_{token}_5m.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        return df

    chunks = []
    cur = from_dt
    while cur < to_dt:
        end = min(cur + timedelta(days=60), to_dt)
        try:
            recs = kite.historical_data(token, cur, end, interval="5minute")
            if recs:
                chunks.append(pd.DataFrame(recs))
        except Exception as e:
            logger.warning(f"5m fetch failed for {symbol} ({cur.date()}–{end.date()}): {e}")
        cur = end + timedelta(days=1)
        time_module.sleep(0.35)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks).drop_duplicates(subset=["date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.rename(columns={"date": "timestamps", "open": "Open", "high": "High",
                              "low": "Low", "close": "Close", "volume": "Volume"})
    df = df.sort_values("timestamps").reset_index(drop=True)
    df.to_parquet(cache_file)
    return df


def get_stock_token(kite, symbol: str, approx_token: int) -> int:
    """Returns verified NSE equity token from instruments list (fallback to approx_token)."""
    cache_file = CACHE_DIR / f"token_{symbol}.txt"
    if cache_file.exists():
        return int(cache_file.read_text().strip())
    try:
        df_inst = pd.DataFrame(kite.instruments("NSE"))
        eq = df_inst[df_inst["segment"] == "NSE"]
        m = eq[eq["tradingsymbol"] == symbol]
        if not m.empty:
            tok = int(m.iloc[0]["instrument_token"])
            cache_file.write_text(str(tok))
            return tok
    except Exception as e:
        logger.warning(f"Token lookup failed for {symbol}: {e}")
    return approx_token


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — DAILY STRATEGY BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def snap_strike(price: float, step: int, direction: str = "round") -> int:
    """Snaps a price to the nearest option strike."""
    if direction == "floor":
        return int(price // step) * step
    elif direction == "ceil":
        return int((price + step - 1) // step) * step
    return round(price / step) * step


def run_daily_backtest(kite) -> None:
    """
    Daily strategy: uses previous N days close prices to predict next-day direction.
    Simulates selling a Put (bullish bias) or Call (bearish bias) at a safe OTM strike.
    Uses Kite's actual daily data for NIFTY 50 for January 2025.
    """
    print("\n" + "=" * 105)
    print("  📈  DAILY STRATEGY BACKTEST — JANUARY 2025 (Kite API Data)")
    print("  Strategy: Previous-close momentum → directional bias → OTM option selling simulation")
    print("=" * 105 + "\n")

    # Fetch NIFTY 50 daily data (need Dec 2024 for lookback)
    print("[INIT] Fetching NIFTY 50 daily data from Kite API (Dec 2024 – Jan 2025)...")
    df = fetch_kite_daily(kite, INDEX_TOKENS["NIFTY"], "NIFTY",
                          datetime(2024, 11, 1), datetime(2025, 1, 31))
    if df.empty:
        print("[ERROR] Could not fetch NIFTY daily data.")
        return

    df = df.set_index("timestamps").sort_index()
    df.index = pd.to_datetime(df.index)

    # Filter Jan 2025 test window
    jan_mask = (df.index >= "2025-01-01") & (df.index <= "2025-01-31")
    jan_dates = df.index[jan_mask]

    if len(jan_dates) == 0:
        print("[ERROR] No Jan 2025 trading days found in Kite data.")
        return

    print(f"[INFO] Jan 2025 trading days found: {len(jan_dates)}")
    print(f"       Range: {jan_dates[0].date()} → {jan_dates[-1].date()}\n")

    print(f"{'DATE':<12} {'PREV CLOSE':<12} {'TODAY CLOSE':<12} {'MOVE %':<10} "
          f"{'BIAS':<8} {'OPTION TRADE':<24} {'RESULT':<15} {'ACTUAL HI':<11} {'ACTUAL LO'}")
    print("-" * 120)

    results = []

    for i, dt in enumerate(jan_dates):
        # Find the previous trading day's close
        prior_dates = df.index[df.index < dt]
        if len(prior_dates) < 5:
            continue
        prev_close = float(df.loc[prior_dates[-1], "Close"])
        prev_5d_closes = df.loc[prior_dates[-5:], "Close"].values.astype(float)

        today_close = float(df.loc[dt, "Close"])
        today_high  = float(df.loc[dt, "High"])
        today_low   = float(df.loc[dt, "Low"])
        today_open  = float(df.loc[dt, "Open"])

        actual_move_pct = (today_close - prev_close) / prev_close * 100.0

        # Bias: 5-day momentum consensus + prev-day to today-open gap
        avg_5d = float(np.mean(prev_5d_closes))
        momentum_signal = (prev_5d_closes[-1] - avg_5d) / avg_5d * 100.0  # % above/below 5d avg
        open_gap = (today_open - prev_close) / prev_close * 100.0

        # Combine: if 5d momentum > 0 and open gap > -0.2% → BULLISH
        score = momentum_signal * 0.6 + open_gap * 0.4
        if score > 0.05:
            bias = "BULLISH"
        elif score < -0.05:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        # Option selling simulation
        if bias == "BULLISH":
            safe_put = snap_strike(prev_close * 0.985, NIFTY_STEP, "floor")  # 1.5% OTM put
            opt_trade = f"SELL {safe_put} PE"
            opt_won = (today_low > safe_put)
        elif bias == "BEARISH":
            safe_call = snap_strike(prev_close * 1.015, NIFTY_STEP, "ceil")  # 1.5% OTM call
            opt_trade = f"SELL {safe_call} CE"
            opt_won = (today_high < safe_call)
        else:
            opt_trade = "NO TRADE"
            opt_won = None

        dir_match = (
            (bias == "BULLISH" and actual_move_pct >= 0) or
            (bias == "BEARISH" and actual_move_pct < 0) or
            bias == "NEUTRAL"
        )

        opt_str = "PROFIT(OTM)" if opt_won is True else ("BREACHED" if opt_won is False else "SKIPPED")
        dir_str = "✓" if dir_match else "✗"

        print(f"{dt.strftime('%Y-%m-%d'):<12} {prev_close:<12.2f} {today_close:<12.2f} "
              f"{actual_move_pct:>+8.2f}%  {bias:<8} {opt_trade:<24} {opt_str:<15} "
              f"{today_high:<11.2f} {today_low:.2f}  {dir_str}")

        results.append({
            "date": dt.strftime("%Y-%m-%d"),
            "bias": bias,
            "prev_close": prev_close,
            "today_close": today_close,
            "actual_move_pct": actual_move_pct,
            "dir_match": dir_match,
            "opt_won": opt_won,
            "opt_trade": opt_trade,
        })

    # Summary
    traded = [r for r in results if r["bias"] != "NEUTRAL"]
    dir_wins = sum(1 for r in traded if r["dir_match"])
    opt_wins = sum(1 for r in traded if r["opt_won"] is True)

    print("\n" + "=" * 65)
    print("  🏆  DAILY STRATEGY — JANUARY 2025 SCORECARD")
    print("=" * 65)
    print(f"  Total trading days tested    : {len(results)}")
    print(f"  Days with directional bias   : {len(traded)} (Neutral: {len(results)-len(traded)})")
    print(f"  Direction accuracy           : {dir_wins}/{len(traded)} "
          f"({dir_wins/len(traded)*100:.1f}%)" if traded else "  Direction accuracy     : N/A")
    print(f"  Option selling wins (OTM)    : {opt_wins}/{len(traded)} "
          f"({opt_wins/len(traded)*100:.1f}%)" if traded else "  Option wins            : N/A")
    bull_days = [r for r in traded if r["bias"] == "BULLISH"]
    bear_days = [r for r in traded if r["bias"] == "BEARISH"]
    print(f"  Bullish days / Bear days     : {len(bull_days)} / {len(bear_days)}")
    avg_move = np.mean([abs(r["actual_move_pct"]) for r in results]) if results else 0
    print(f"  Avg absolute daily move      : {avg_move:.2f}%")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — SECTOR MOMENTUM STRATEGY BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_costs(pnl_pts: float, entry_price: float) -> float:
    net = pnl_pts - entry_price * COST_BPS_ROUNDTRIP / 10_000.0
    net -= entry_price * BREAKOUT_SLIPPAGE_PCT / 100.0
    return net


def _get_day_5m(df_5m: pd.DataFrame, symbol: str, date_obj: date) -> pd.DataFrame:
    """Slices a day's 5m candles for a symbol from the consolidated 5m dataframe."""
    try:
        mask = df_5m.index.date == date_obj
        df_day = df_5m.loc[mask]
        if df_day.empty:
            return pd.DataFrame()
        if isinstance(df_day.columns, pd.MultiIndex):
            return pd.DataFrame({
                "Open":   df_day[("Open",   symbol)],
                "High":   df_day[("High",   symbol)],
                "Low":    df_day[("Low",    symbol)],
                "Close":  df_day[("Close",  symbol)],
                "Volume": df_day[("Volume", symbol)],
            }).dropna()
        else:
            return df_day[[c for c in ["Open","High","Low","Close","Volume"] if c in df_day.columns]].dropna()
    except Exception:
        return pd.DataFrame()


def _simulate_stock_day_momentum(
    sym: str, stk_5m: pd.DataFrame, nifty_5m: pd.DataFrame,
    direction: str, bias: str, pdh: float, pdl: float, orh15: float, orl15: float,
    current_date: date
) -> Optional[dict]:
    """Runs the full Phase 2 candle-by-candle momentum simulation for one stock."""

    session = stk_5m[
        (stk_5m.index.time >= time(9, 30)) & (stk_5m.index.time <= time(15, 15))
    ]
    if session.empty:
        return None

    ema10 = (
        session["Close"].ewm(span=EMA_TRAIL_PERIOD, adjust=False).mean()
        if USE_EMA_TRAILING else None
    )

    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    entry_time_str = ""
    entry_ts = None
    pullback = None
    min_vol = float("inf")

    candle_times = list(session.index)

    for i, c_ts in enumerate(candle_times):
        c = session.loc[c_ts]
        c_open, c_close = float(c["Open"]), float(c["Close"])
        c_high, c_low, c_vol = float(c["High"]), float(c["Low"]), float(c["Volume"])

        if not in_trade:
            # Gate 1: Identify pullback candle (opposite colour, lowest volume so far)
            is_opp = (c_close < c_open) if bias == "BULLISH" else (c_close > c_open)
            if is_opp and c_vol < min_vol:
                pullback = {"ts": c_ts, "high": c_high, "low": c_low, "vol": c_vol}
                min_vol = c_vol

            # Gate 2: Breakout of pullback candle
            if pullback and c_ts > pullback["ts"]:
                fired = (c_high > pullback["high"]) if bias == "BULLISH" else (c_low < pullback["low"])
                if fired:
                    # Gate 3: Nifty realignment
                    nifty_aligned = False
                    if c_ts in nifty_5m.index:
                        nc = nifty_5m.loc[c_ts]
                        nifty_aligned = (
                            (float(nc["Close"]) > float(nc["Open"])) if bias == "BULLISH"
                            else (float(nc["Close"]) < float(nc["Open"]))
                        )
                    if not nifty_aligned:
                        continue

                    # Gate 4: PDH/ORH structural break
                    if bias == "BULLISH":
                        ok = pdh > 0 and orh15 > 0 and c_high > pdh and c_high > orh15
                    else:
                        ok = pdl > 0 and orl15 > 0 and c_low < pdl and c_low < orl15
                    if not ok:
                        continue

                    # Enter trade
                    in_trade = True
                    entry_ts = c_ts
                    entry_time_str = c_ts.strftime("%H:%M")
                    if bias == "BULLISH":
                        entry_price = pullback["high"]
                        stop_loss   = pullback["low"] * (1.0 - SL_BUFFER)
                    else:
                        entry_price = pullback["low"]
                        stop_loss   = pullback["high"] * (1.0 + SL_BUFFER)

        if in_trade:
            # Exit 1: Hard stop loss
            sl_hit = (c_low <= stop_loss) if bias == "BULLISH" else (c_high >= stop_loss)
            if sl_hit:
                exit_price = stop_loss
                pnl = _apply_costs(
                    (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price),
                    entry_price
                )
                return _make_trade_dict(sym, current_date, direction, entry_time_str, entry_price,
                                        stop_loss, c_ts, exit_price, "STOP_LOSS", pnl)

            # Exit 2: 10 EMA trailing
            if USE_EMA_TRAILING and entry_ts and c_ts > entry_ts and ema10 is not None:
                loc = ema10.index.get_loc(c_ts)
                if loc + 1 >= EMA_TRAIL_PERIOD:
                    ema_val = float(ema10.iloc[loc])
                    trail_hit = (c_close < ema_val) if bias == "BULLISH" else (c_close > ema_val)
                    if trail_hit:
                        exit_price = c_close
                        pnl = _apply_costs(
                            (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price),
                            entry_price
                        )
                        return _make_trade_dict(sym, current_date, direction, entry_time_str, entry_price,
                                                stop_loss, c_ts, exit_price, "10_EMA_TRAIL", pnl)

            # Exit 3: 15:15 square-off
            if c_ts.time() >= time(15, 15) or i == len(candle_times) - 1:
                exit_price = c_close
                pnl = _apply_costs(
                    (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price),
                    entry_price
                )
                return _make_trade_dict(sym, current_date, direction, entry_time_str, entry_price,
                                        stop_loss, c_ts, exit_price, "SQUARE_OFF_1515", pnl)

    return None


def _make_trade_dict(sym, dt, direction, entry_time, entry_price, sl, exit_ts, exit_price, reason, pnl_pts):
    pnl_pct = (pnl_pts / entry_price) * 100.0
    risk = abs(entry_price - sl)
    rr = pnl_pts / risk if risk > 0 else 0.0
    return {
        "date":         dt.strftime("%Y-%m-%d"),
        "symbol":       sym,
        "direction":    direction,
        "entry_time":   entry_time,
        "entry_price":  round(entry_price, 2),
        "stop_loss":    round(sl, 2),
        "exit_time":    exit_ts.strftime("%H:%M"),
        "exit_price":   round(exit_price, 2),
        "exit_reason":  reason,
        "pnl_pts":      round(pnl_pts, 2),
        "pnl_pct":      round(pnl_pct, 2),
        "risk_pts":     round(risk, 2),
        "rr_achieved":  round(rr, 2),
    }


def run_momentum_backtest(kite) -> None:
    """
    Sector Momentum Strategy backtest for January 2025.
    Fetches 5m and daily data from Kite API for all 57 instruments.
    """
    print("\n" + "=" * 105)
    print("  🚀  SECTOR MOMENTUM STRATEGY BACKTEST — JANUARY 2025 (Kite API Data)")
    print("  Phase 1: 09:30 Nifty bias + sector rank + top-3 stock selection")
    print("  Phase 2: Pullback candle → Breakout → Nifty realign → PDH/ORH gate → Trade simulation")
    print("=" * 105 + "\n")

    print("[INIT] Fetching instrument tokens from Kite NSE instruments list...")
    token_map: dict[str, int] = dict(INDEX_TOKENS)  # sector indices already mapped

    # Resolve stock tokens
    df_inst = pd.DataFrame(kite.instruments("NSE"))
    eq = df_inst[df_inst["segment"] == "NSE"]
    for sector, stocks in SECTOR_STOCKS.items():
        for _yf, nse_sym, approx_tok in stocks:
            if nse_sym not in token_map:
                m = eq[eq["tradingsymbol"] == nse_sym]
                if not m.empty:
                    token_map[nse_sym] = int(m.iloc[0]["instrument_token"])
                else:
                    token_map[nse_sym] = approx_tok

    all_symbols = list(token_map.keys())
    print(f"[INIT] {len(all_symbols)} instruments resolved. Fetching Jan 2025 historical data...")
    print("       (Data will be cached locally after first download)\n")

    # Download 5m and daily data for Dec 2024 – Jan 2025
    from_dt    = datetime(2024, 12, 1)
    to_dt      = datetime(2025, 1, 31)
    daily_from = datetime(2024, 11, 1)

    data_5m: dict[str, pd.DataFrame] = {}
    data_daily: dict[str, pd.DataFrame] = {}

    total = len(all_symbols)
    for i, (sym, tok) in enumerate(token_map.items(), 1):
        print(f"\r  Downloading [{i:>2}/{total}] {sym:<15}", end="", flush=True)
        data_5m[sym]    = fetch_kite_5m(kite, tok, sym, from_dt, to_dt)
        data_daily[sym] = fetch_kite_daily(kite, tok, sym, daily_from, to_dt)

    print(f"\n[INIT] All data fetched. Starting backtest...\n")

    # Build unified 5m index
    all_ts_5m = sorted(set(
        ts for df in data_5m.values() if not df.empty
        for ts in df["timestamps"].tolist()
    ))
    idx_5m = pd.DatetimeIndex(all_ts_5m)

    # Build multi-index 5m DataFrame indexed by timestamp
    frames_5m: dict[str, pd.DataFrame] = {}
    for sym, df in data_5m.items():
        if not df.empty:
            frames_5m[sym] = df.set_index("timestamps")[["Open","High","Low","Close","Volume"]]

    # Get Jan 2025 trading days from NIFTY 5m data
    nifty_5m_raw = frames_5m.get("NIFTY", pd.DataFrame())
    if nifty_5m_raw.empty:
        print("[ERROR] No NIFTY 5m data available.")
        return

    jan_trading_days = sorted(set(
        d for d in nifty_5m_raw.index.date
        if d >= date(2025, 1, 1) and d <= date(2025, 1, 31)
    ))
    all_daily_days = sorted(set(
        d for df in data_daily.values() if not df.empty
        for d in pd.to_datetime(df["timestamps"]).dt.date.tolist()
    ))

    print(f"[INFO] Sector Momentum trading days in Jan 2025: {len(jan_trading_days)}")
    print(f"       {jan_trading_days[0]} → {jan_trading_days[-1]}\n")

    # ── Backtest loop ─────────────────────────────────────────────────────────
    day_results = []
    all_trades  = []

    print(f"{'DATE':<12} {'BIAS':<8} {'NIFTY RET':<11} {'WINNING SECTOR':<16} {'TOP 3 STOCKS':<32} {'TRADES'}")
    print("-" * 95)

    for current_date in jan_trading_days:
        date_str = current_date.strftime("%Y-%m-%d")

        # Prior trading day for PDH/PDL
        prior_days = [d for d in all_daily_days if d < current_date]
        if not prior_days:
            continue
        prev_date = prior_days[-1]

        # Get NIFTY 5m candles
        nifty_today = frames_5m.get("NIFTY", pd.DataFrame())
        nifty_day = nifty_today[nifty_today.index.date == current_date] if not nifty_today.empty else pd.DataFrame()

        if nifty_day.empty or len(nifty_day) < 3:
            print(f"{date_str:<12} NO_DATA")
            continue

        # Nifty bias: first 3 candles (09:15, 09:20, 09:25) before 09:30
        nifty_pre930 = nifty_day[nifty_day.index.time < time(9, 30)]
        if len(nifty_pre930) < 3:
            print(f"{date_str:<12} SKIP (insufficient Nifty pre-open candles)")
            continue

        nifty_open = float(nifty_pre930["Open"].iloc[0])
        nifty_930  = float(nifty_pre930["Close"].iloc[-1])
        nifty_ret  = (nifty_930 - nifty_open) / nifty_open

        bias = (
            "BULLISH" if nifty_ret > BIAS_THRESHOLD else
            "BEARISH" if nifty_ret < -BIAS_THRESHOLD else "NEUTRAL"
        )

        if bias == "NEUTRAL":
            day_results.append({"date": date_str, "bias": "NEUTRAL", "nifty_ret": nifty_ret,
                                 "sector": "", "stocks": [], "trades": []})
            print(f"{date_str:<12} {'NEUTRAL':<8} {nifty_ret*100:>+6.2f}%     {'—':<16} {'—':<32} 0")
            continue

        # Sector ranking at 09:30
        sector_ranks = []
        for sect_name, sect_sym in [
            ("IT","IT"), ("Bank","Bank"), ("Auto","Auto"), ("Pharma","Pharma"),
            ("FMCG","FMCG"), ("Metal","Metal"), ("Realty","Realty"), ("Energy","Energy")
        ]:
            # Get sector prev close from daily data
            sect_daily = data_daily.get(sect_sym, pd.DataFrame())
            if sect_daily.empty:
                continue
            sect_daily_idx = sect_daily.set_index("timestamps")
            prev_rows = sect_daily_idx[sect_daily_idx.index.date == prev_date]
            if prev_rows.empty:
                continue
            sect_prev_close = float(prev_rows["Close"].iloc[0])

            sect_5m = frames_5m.get(sect_sym, pd.DataFrame())
            if sect_5m.empty:
                continue
            sect_day = sect_5m[sect_5m.index.date == current_date]
            sect_pre = sect_day[sect_day.index.time < time(9, 30)]
            if sect_pre.empty:
                continue
            sect_price_930 = float(sect_pre["Close"].iloc[-1])
            sect_ret = (sect_price_930 - sect_prev_close) / sect_prev_close
            sector_ranks.append({"sector": sect_name, "return": sect_ret, "sym": sect_sym})

        if not sector_ranks:
            print(f"{date_str:<12} {bias:<8} {nifty_ret*100:>+6.2f}%     {'NO SECTORS':<16} {'—':<32} 0")
            continue

        sector_ranks.sort(key=lambda x: x["return"], reverse=True)
        winning = sector_ranks[0] if bias == "BULLISH" else sector_ranks[-1]
        winning_sector = winning["sector"]
        winning_ret    = winning["return"]

        # Stock selection within winning sector
        candidates = []
        for _yf, nse_sym, _tok in SECTOR_STOCKS.get(winning_sector, []):
            stk_daily = data_daily.get(nse_sym, pd.DataFrame())
            if stk_daily.empty:
                continue
            stk_daily_idx = stk_daily.set_index("timestamps")
            prev_stk = stk_daily_idx[stk_daily_idx.index.date == prev_date]
            if prev_stk.empty:
                continue

            stk_prev_close = float(prev_stk["Close"].iloc[0])
            stk_pdh        = float(prev_stk["High"].iloc[0])
            stk_pdl        = float(prev_stk["Low"].iloc[0])

            stk_5m = frames_5m.get(nse_sym, pd.DataFrame())
            if stk_5m.empty:
                continue
            stk_day = stk_5m[stk_5m.index.date == current_date]
            stk_pre = stk_day[stk_day.index.time < time(9, 30)]
            if len(stk_pre) < 3:
                continue

            stk_price_930 = float(stk_pre["Close"].iloc[-1])
            stk_vol_930   = float(stk_pre["Volume"].iloc[-1])
            stk_ret       = (stk_price_930 - stk_prev_close) / stk_prev_close
            stk_orh15     = float(stk_pre["High"].max())
            stk_orl15     = float(stk_pre["Low"].min())

            candidates.append({
                "sym": nse_sym, "return": stk_ret, "volume": max(stk_vol_930, 1.0),
                "pdh": stk_pdh, "pdl": stk_pdl, "orh15": stk_orh15, "orl15": stk_orl15,
            })

        if not candidates:
            print(f"{date_str:<12} {bias:<8} {nifty_ret*100:>+6.2f}%     {winning_sector:<16} {'NO CANDS':<32} 0")
            continue

        total_vol = sum(c["volume"] for c in candidates)
        for c in candidates:
            vw = c["volume"] / total_vol
            c["score"] = c["return"] * (1.0 + vw)

        candidates.sort(key=lambda x: x["score"], reverse=(bias == "BULLISH"))
        top_stocks = candidates[:NUM_TOP_STOCKS]
        top_names  = [c["sym"] for c in top_stocks]

        # Phase 2: simulate each stock
        trades_today = []
        for stock in top_stocks:
            stk_5m_df = frames_5m.get(stock["sym"], pd.DataFrame())
            if stk_5m_df.empty:
                continue
            stk_day_5m = stk_5m_df[stk_5m_df.index.date == current_date]
            if stk_day_5m.empty:
                continue

            trade = _simulate_stock_day_momentum(
                sym=stock["sym"],
                stk_5m=stk_day_5m,
                nifty_5m=nifty_day,
                direction="LONG" if bias == "BULLISH" else "SHORT",
                bias=bias,
                pdh=stock["pdh"],
                pdl=stock["pdl"],
                orh15=stock["orh15"],
                orl15=stock["orl15"],
                current_date=current_date,
            )
            if trade:
                trade["sector"] = winning_sector
                trades_today.append(trade)
                all_trades.append(trade)

        stocks_str = ", ".join(top_names)
        print(f"{date_str:<12} {bias:<8} {nifty_ret*100:>+6.2f}%     {winning_sector:<16} {stocks_str:<32} {len(trades_today)}")

        day_results.append({
            "date": date_str, "bias": bias, "nifty_ret": nifty_ret,
            "sector": winning_sector, "stocks": top_names, "trades": trades_today,
        })

    # ── Trade detail table ────────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("TRADE EXECUTION DETAIL:")
    print("=" * 130)
    header = (f"{'DATE':<12} {'SYMBOL':<12} {'DIR':<6} {'ENTRY':<6} {'PRICE':>9} {'SL':>9} "
              f"{'EXIT':<6} {'EXIT PX':>9} {'REASON':<16} {'PNL PTS':>9} {'PNL %':>8} {'R:R':>6}")
    print(header)
    print("-" * 130)

    for t in all_trades:
        print(f"{t['date']:<12} {t['symbol']:<12} {t['direction']:<6} {t['entry_time']:<6} "
              f"{t['entry_price']:>9.2f} {t['stop_loss']:>9.2f} {t['exit_time']:<6} "
              f"{t['exit_price']:>9.2f} {t['exit_reason']:<16} {t['pnl_pts']:>+9.2f} "
              f"{t['pnl_pct']:>+7.2f}% {t['rr_achieved']:>+6.2f}")

    print("=" * 130)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if all_trades:
        csv_path = CACHE_DIR / "momentum_backtest_jan2025.csv"
        pd.DataFrame(all_trades).to_csv(csv_path, index=False)
        print(f"\n[SAVED] {len(all_trades)} trades written to {csv_path}")

    # ── Performance metrics ───────────────────────────────────────────────────
    wins   = [t for t in all_trades if t["pnl_pct"] > 0]
    losses = [t for t in all_trades if t["pnl_pct"] <= 0]
    win_rate = (len(wins) / len(all_trades) * 100.0) if all_trades else 0.0
    total_pnl_pct = sum(t["pnl_pct"] for t in all_trades)
    gross_profit  = sum(t["pnl_pct"] for t in wins)
    gross_loss    = abs(sum(t["pnl_pct"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
    avg_win_pct   = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
    avg_loss_pct  = np.mean([t["pnl_pct"] for t in losses]) if losses else 0.0
    sl_exits      = sum(1 for t in all_trades if t["exit_reason"] == "STOP_LOSS")
    ema_exits     = sum(1 for t in all_trades if t["exit_reason"] == "10_EMA_TRAIL")
    sq_exits      = sum(1 for t in all_trades if t["exit_reason"] == "SQUARE_OFF_1515")

    biased_days = len([r for r in day_results if r["bias"] != "NEUTRAL"])
    neutral_days = len([r for r in day_results if r["bias"] == "NEUTRAL"])

    print("\n" + "=" * 65)
    print("  🏆  SECTOR MOMENTUM — JANUARY 2025 PERFORMANCE METRICS")
    print("=" * 65)
    print(f"  Total Jan 2025 trading days  : {len(jan_trading_days)}")
    print(f"  Days with directional bias   : {biased_days}  (Neutral: {neutral_days})")
    print(f"  Total trades executed        : {len(all_trades)}")
    print(f"  Winning trades               : {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losing trades                : {len(losses)} ({100-win_rate:.1f}%)")
    print(f"  Avg win  / Avg loss          : {avg_win_pct:+.2f}% / {avg_loss_pct:+.2f}%")
    print(f"  Gross profit                 : {gross_profit:+.2f}%")
    print(f"  Gross loss                   : -{gross_loss:.2f}%")
    print(f"  Total cumulative P&L         : {total_pnl_pct:+.2f}%")
    print(f"  Profit factor                : {profit_factor:.2f}")
    print(f"  Exit breakdown               : SL={sl_exits}  EMA-trail={ema_exits}  SquareOff={sq_exits}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 105)
    print("  JANUARY 2025 BACKTEST SUITE — Zerodha Kite API Data")
    print("  Strategy 1: Daily Direction + OTM Option Selling  |  Strategy 2: Sector Momentum Intraday")
    print("=" * 105)
    print(f"\n  Cache directory: {CACHE_DIR}")

    kite = get_kite()

    # ── Run Strategy 1: Daily Backtest ────────────────────────────────────────
    run_daily_backtest(kite)

    # ── Run Strategy 2: Sector Momentum Backtest ──────────────────────────────
    run_momentum_backtest(kite)

    print("\n✅  All backtests complete.")
