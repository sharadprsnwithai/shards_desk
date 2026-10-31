"""
sector_momentum_desk/backtester.py
1-Month Historical Backtesting Engine for the Sector Momentum Strategy.

Replays Phase 1 (Nifty bias + Sector + Stock Selection at 09:30 AM)
and Phase 2 (Lowest-volume pullback candle, breakout, Nifty realignment, PDH/ORH structural break, SL/Square-off exits).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sector_momentum_desk.config import (
    BIAS_THRESHOLD,
    DATA_CACHE_DIR,
    EMA_TRAIL_PERIOD,
    NUM_TOP_STOCKS,
    SECTOR_STOCKS,
    SECTOR_TICKERS,
    SL_BUFFER,
    USE_EMA_TRAILING,
)

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_CACHE_DIR / "backtest_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Trade:
    date: str
    symbol: str
    direction: str
    sector: str
    entry_time: str
    entry_price: float
    stop_loss: float
    exit_time: str
    exit_price: float
    exit_reason: str       # "STOP_LOSS" | "SQUARE_OFF_1515"
    pnl_pts: float
    pnl_pct: float
    risk_pts: float
    rr_achieved: float


@dataclass
class DayResult:
    date: str
    nifty_bias: str
    nifty_return: float
    winning_sector: str
    sector_return: float
    top_stocks: list[str]
    trades: list[Trade] = field(default_factory=list)


class SectorMomentumBacktester:
    """
    High-speed, deterministic backtest engine for Sector Momentum Strategy.
    Supports both official Zerodha Kite API historical data ('zerodha') and 'yfinance'.
    """

    def __init__(self, period: str = "6mo", source: str = "zerodha"):
        self.period = period
        self.source = source
        self.df_5m: Optional[pd.DataFrame] = None
        self.df_daily: Optional[pd.DataFrame] = None
        self._all_tickers = (
            ["^NSEI"]
            + list(SECTOR_TICKERS.values())
            + [t for s_list in SECTOR_STOCKS.values() for t, _ in s_list]
        )

    def load_data(self, force_download: bool = False) -> None:
        """Download or load cached 5m and daily multi-ticker dataframes."""
        if self.source == "zerodha":
            from sector_momentum_desk.zerodha_data import fetch_and_cache_zerodha_data
            days = 180 if self.period in ("6mo", "180d") else 30
            self.df_5m, self.df_daily = fetch_and_cache_zerodha_data(days=days, force=force_download)
            logger.info(f"[ZerodhaBacktest] Loaded 5m shape: {self.df_5m.shape}, daily shape: {self.df_daily.shape}")
            return

        file_5m = CACHE_DIR / f"data_5m_{self.period}.parquet"
        file_daily = CACHE_DIR / f"data_daily_{self.period}.parquet"

        if not force_download and file_5m.exists() and file_daily.exists():
            logger.info("Loading backtest data from local parquet cache...")
            self.df_5m = pd.read_parquet(file_5m)
            self.df_daily = pd.read_parquet(file_daily)
            logger.info(f"Loaded 5m shape: {self.df_5m.shape}, daily shape: {self.df_daily.shape}")
            return

        logger.info(f"Downloading {len(self._all_tickers)} tickers for period='{self.period}', interval='5m'...")
        df_5m = yf.download(self._all_tickers, period=self.period, interval="5m", progress=False, auto_adjust=True)
        logger.info(f"Downloading daily data for period='3mo'...")
        df_daily = yf.download(self._all_tickers, period="3mo", interval="1d", progress=False, auto_adjust=True)

        # Convert timestamps to IST (Asia/Kolkata) then drop tz
        if df_5m.index.tz is not None:
            df_5m.index = df_5m.index.tz_convert("Asia/Kolkata").tz_localize(None)
        else:
            df_5m.index = pd.to_datetime(df_5m.index)

        if df_daily.index.tz is not None:
            df_daily.index = df_daily.index.tz_convert("Asia/Kolkata").tz_localize(None)
        else:
            df_daily.index = pd.to_datetime(df_daily.index)

        # Cache locally
        df_5m.to_parquet(file_5m)
        df_daily.to_parquet(file_daily)

        self.df_5m = df_5m
        self.df_daily = df_daily
        logger.info(f"Cached backtest data to {CACHE_DIR}")

    def run(self) -> list[DayResult]:
        """Runs the backtest across all trading days present in the dataset."""
        if self.df_5m is None or self.df_daily is None:
            self.load_data()

        # Find unique trading days in 5m dataset
        all_dates = sorted(list(set(self.df_5m.index.date)))
        logger.info(f"Found {len(all_dates)} trading days in dataset: {all_dates[0]} to {all_dates[-1]}")

        # Need previous day daily data for PDH/PDL and sector prev close
        daily_dates = sorted(list(set(self.df_daily.index.date)))

        results: list[DayResult] = []

        for current_date in all_dates:
            # Find previous trading day in daily dataset
            prior_daily_dates = [d for d in daily_dates if d < current_date]
            if not prior_daily_dates:
                logger.debug(f"Skipping {current_date}: no prior daily data.")
                continue
            prev_date = prior_daily_dates[-1]

            day_res = self._backtest_single_day(current_date, prev_date)
            results.append(day_res)

        return results

    def _backtest_single_day(self, current_date, prev_date) -> DayResult:
        """Simulates Phase 1 and Phase 2 for a single trading day."""
        date_str = current_date.strftime("%Y-%m-%d")

        # â”€â”€ 1. Nifty 50 Direction Check at 09:30 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Get today's 5m candles for ^NSEI
        nifty_today = self._get_ticker_day_data("^NSEI", current_date)
        if nifty_today.empty or len(nifty_today) < 3:
            return DayResult(date=date_str, nifty_bias="NO_DATA", nifty_return=0.0,
                             winning_sector="", sector_return=0.0, top_stocks=[])

        # Candles up to 09:30 AM (09:15, 09:20, 09:25)
        nifty_930 = nifty_today[nifty_today.index.time < time(9, 30)]
        if len(nifty_930) < 3:
            return DayResult(date=date_str, nifty_bias="NO_DATA", nifty_return=0.0,
                             winning_sector="", sector_return=0.0, top_stocks=[])

        nifty_open_915 = float(nifty_930["Open"].iloc[0])
        nifty_close_930 = float(nifty_930["Close"].iloc[-1])
        nifty_ret = (nifty_close_930 - nifty_open_915) / nifty_open_915

        if nifty_ret > BIAS_THRESHOLD:
            bias = "BULLISH"
        elif nifty_ret < -BIAS_THRESHOLD:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
            return DayResult(date=date_str, nifty_bias="NEUTRAL", nifty_return=nifty_ret,
                             winning_sector="", sector_return=0.0, top_stocks=[])

        # â”€â”€ 2. Sector Ranking at 09:30 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sector_ranks = []
        for sect_name, ticker in SECTOR_TICKERS.items():
            sym_key = sect_name if self.source == "zerodha" else ticker
            sect_prev_close = self._get_daily_close(sym_key, prev_date)
            sect_today = self._get_ticker_day_data(sym_key, current_date)
            if sect_prev_close is None or sect_today.empty:
                continue
            sect_930 = sect_today[sect_today.index.time < time(9, 30)]
            if sect_930.empty:
                continue
            sect_price_930 = float(sect_930["Close"].iloc[-1])
            sect_ret = (sect_price_930 - sect_prev_close) / sect_prev_close
            sector_ranks.append({"sector": sect_name, "return": sect_ret})

        if not sector_ranks:
            return DayResult(date=date_str, nifty_bias=bias, nifty_return=nifty_ret,
                             winning_sector="NONE", sector_return=0.0, top_stocks=[])

        sector_ranks.sort(key=lambda x: x["return"], reverse=True)
        winning_sector_entry = sector_ranks[0] if bias == "BULLISH" else sector_ranks[-1]
        winning_sector = winning_sector_entry["sector"]
        winning_sector_ret = winning_sector_entry["return"]

        # â”€â”€ 3. Stock Scoring within Winning Sector â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        stock_candidates = []
        for yf_tick, nse_sym in SECTOR_STOCKS.get(winning_sector, []):
            sym_key = nse_sym if self.source == "zerodha" else yf_tick
            stk_prev_close = self._get_daily_close(sym_key, prev_date)
            stk_pdh = self._get_daily_high(sym_key, prev_date)
            stk_pdl = self._get_daily_low(sym_key, prev_date)

            stk_today = self._get_ticker_day_data(sym_key, current_date)
            if stk_prev_close is None or stk_today.empty:
                continue

            stk_930 = stk_today[stk_today.index.time < time(9, 30)]
            if len(stk_930) < 3:
                continue

            stk_price_930 = float(stk_930["Close"].iloc[-1])
            stk_vol_930 = float(stk_930["Volume"].iloc[-1])
            stk_ret = (stk_price_930 - stk_prev_close) / stk_prev_close
            stk_orh15 = float(stk_930["High"].max())
            stk_orl15 = float(stk_930["Low"].min())

            stock_candidates.append({
                "ticker": sym_key,
                "nse_sym": nse_sym,
                "return": stk_ret,
                "volume": max(stk_vol_930, 1.0),
                "pdh": stk_pdh or 0.0,
                "pdl": stk_pdl or 0.0,
                "orh15": stk_orh15,
                "orl15": stk_orl15,
            })

        if not stock_candidates:
            return DayResult(date=date_str, nifty_bias=bias, nifty_return=nifty_ret,
                             winning_sector=winning_sector, sector_return=winning_sector_ret,
                             top_stocks=[])

        total_vol = sum(c["volume"] for c in stock_candidates)
        for c in stock_candidates:
            oi_w = c["volume"] / total_vol if total_vol > 0 else (1.0 / len(stock_candidates))
            c["score"] = c["return"] * (1.0 + oi_w)

        if bias == "BULLISH":
            stock_candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            stock_candidates.sort(key=lambda x: x["score"], reverse=False)

        top_stocks = stock_candidates[:NUM_TOP_STOCKS]
        top_symbols = [c["nse_sym"] for c in top_stocks]

        # â”€â”€ 4. Phase 2: Candle-by-Candle Simulation (09:30 to 15:15) â”€â”€â”€â”€â”€â”€â”€â”€â”€
        trades: list[Trade] = []

        for stock in top_stocks:
            trade = self._simulate_stock_day(
                ticker=stock["ticker"],
                nse_sym=stock["nse_sym"],
                direction="LONG" if bias == "BULLISH" else "SHORT",
                bias=bias,
                pdh=stock["pdh"],
                pdl=stock["pdl"],
                orh15=stock["orh15"],
                orl15=stock["orl15"],
                sector=winning_sector,
                current_date=current_date,
                nifty_today=nifty_today,
            )
            if trade:
                trades.append(trade)

        return DayResult(
            date=date_str,
            nifty_bias=bias,
            nifty_return=nifty_ret,
            winning_sector=winning_sector,
            sector_return=winning_sector_ret,
            top_stocks=top_symbols,
            trades=trades,
        )

    def _simulate_stock_day(
        self, ticker, nse_sym, direction, bias, pdh, pdl, orh15, orl15,
        sector, current_date, nifty_today
    ) -> Optional[Trade]:
        """Simulates Phase 2 intraday trigger and trade execution for one stock."""
        stk_today = self._get_ticker_day_data(ticker, current_date)
        if stk_today.empty:
            return None

        # Intraday candles starting from 09:30 AM up to 15:15 PM
        session_candles = stk_today[
            (stk_today.index.time >= time(9, 30)) & (stk_today.index.time <= time(15, 15))
        ]
        if session_candles.empty:
            return None

        pullback_candle = None
        current_min_vol = float("inf")

        # Per-session 10-EMA (live parity: entry_trigger computes the trail EMA on
        # the trading day's candles only, never across the overnight gap)
        ema10_series = (
            session_candles["Close"].ewm(span=EMA_TRAIL_PERIOD, adjust=False).mean()
            if USE_EMA_TRAILING else None
        )

        in_trade = False
        entry_price = 0.0
        stop_loss = 0.0
        entry_time_str = ""
        entry_ts = None

        candle_times = list(session_candles.index)

        for i, c_ts in enumerate(candle_times):
            c = session_candles.loc[c_ts]
            c_open = float(c["Open"])
            c_close = float(c["Close"])
            c_high = float(c["High"])
            c_low = float(c["Low"])
            c_vol = float(c["Volume"])

            if not in_trade:
                # â”€â”€ Gate 1: Check if candle is opposite direction & lowest volume â”€â”€
                is_opposite = (c_close < c_open) if bias == "BULLISH" else (c_close > c_open)
                if is_opposite and c_vol < current_min_vol:
                    pullback_candle = {
                        "ts": c_ts,
                        "open": c_open, "close": c_close,
                        "high": c_high, "low": c_low,
                        "vol": c_vol,
                    }
                    current_min_vol = c_vol

                # â”€â”€ Gate 2: Breakout of existing pullback candle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if pullback_candle is not None and c_ts > pullback_candle["ts"]:
                    breakout_fired = (
                        c_high > pullback_candle["high"]
                        if bias == "BULLISH"
                        else c_low < pullback_candle["low"]
                    )
                    if breakout_fired:
                        # â”€â”€ Gate 3: Nifty Realignment Check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        if c_ts in nifty_today.index:
                            nifty_c = nifty_today.loc[c_ts]
                            nifty_open = float(nifty_c["Open"])
                            nifty_close = float(nifty_c["Close"])
                            nifty_aligned = (
                                (nifty_close > nifty_open)
                                if bias == "BULLISH"
                                else (nifty_close < nifty_open)
                            )
                        else:
                            nifty_aligned = False

                        if not nifty_aligned:
                            continue

                        # â”€â”€ Gate 4: Structural Breakout Check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        if bias == "BULLISH":
                            structural_pass = (
                                pdh > 0 and orh15 > 0 and c_high > pdh and c_high > orh15
                            )
                        else:
                            structural_pass = (
                                pdl > 0 and orl15 > 0 and c_low < pdl and c_low < orl15
                            )

                        if not structural_pass:
                            continue

                        # â”€â”€ ALL 4 GATES PASSED: ENTER TRADE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        in_trade = True
                        entry_ts = c_ts
                        entry_time_str = c_ts.strftime("%H:%M")
                        if bias == "BULLISH":
                            entry_price = pullback_candle["high"]
                            stop_loss = pullback_candle["low"] * (1.0 - SL_BUFFER)
                        else:
                            entry_price = pullback_candle["low"]
                            stop_loss = pullback_candle["high"] * (1.0 + SL_BUFFER)

            # â”€â”€ If in trade, check exit conditions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if in_trade:
                # 1. Check Hard Stop Loss hit
                sl_hit = (c_low <= stop_loss) if bias == "BULLISH" else (c_high >= stop_loss)
                if sl_hit:
                    exit_price = stop_loss
                    pnl_pts = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
                    pnl_pct = (pnl_pts / entry_price) * 100.0
                    risk_pts = abs(entry_price - stop_loss)
                    rr = pnl_pts / risk_pts if risk_pts > 0 else 0.0

                    return Trade(
                        date=current_date.strftime("%Y-%m-%d"),
                        symbol=nse_sym,
                        direction=direction,
                        sector=sector,
                        entry_time=entry_time_str,
                        entry_price=round(entry_price, 2),
                        stop_loss=round(stop_loss, 2),
                        exit_time=c_ts.strftime("%H:%M"),
                        exit_price=round(exit_price, 2),
                        exit_reason="STOP_LOSS",
                        pnl_pts=round(pnl_pts, 2),
                        pnl_pct=round(pnl_pct, 2),
                        risk_pts=round(risk_pts, 2),
                        rr_achieved=round(rr, 2),
                    )

                # 2. Check Dynamic 10 EMA Trailing Exit (on candles after entry)
                if USE_EMA_TRAILING and entry_ts is not None and c_ts > entry_ts and ema10_series is not None:
                    ema_val = None
                    loc = ema10_series.index.get_loc(c_ts)
                    if loc + 1 >= EMA_TRAIL_PERIOD:   # live guard: len(candles) >= period
                        ema_val = float(ema10_series.iloc[loc])

                    if ema_val is not None:
                        ema_trail_triggered = (
                            (c_close < ema_val) if bias == "BULLISH" else (c_close > ema_val)
                        )
                        if ema_trail_triggered:
                            exit_price = c_close
                            pnl_pts = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
                            pnl_pct = (pnl_pts / entry_price) * 100.0
                            risk_pts = abs(entry_price - stop_loss)
                            rr = pnl_pts / risk_pts if risk_pts > 0 else 0.0

                            return Trade(
                                date=current_date.strftime("%Y-%m-%d"),
                                symbol=nse_sym,
                                direction=direction,
                                sector=sector,
                                entry_time=entry_time_str,
                                entry_price=round(entry_price, 2),
                                stop_loss=round(stop_loss, 2),
                                exit_time=c_ts.strftime("%H:%M"),
                                exit_price=round(exit_price, 2),
                                exit_reason="10_EMA_TRAIL",
                                pnl_pts=round(pnl_pts, 2),
                                pnl_pct=round(pnl_pct, 2),
                                risk_pts=round(risk_pts, 2),
                                rr_achieved=round(rr, 2),
                            )

                # 3. Check 15:15 PM Square-off (end of session)
                if c_ts.time() >= time(15, 15) or i == len(candle_times) - 1:
                    exit_price = c_close
                    pnl_pts = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
                    pnl_pct = (pnl_pts / entry_price) * 100.0
                    risk_pts = abs(entry_price - stop_loss)
                    rr = pnl_pts / risk_pts if risk_pts > 0 else 0.0

                    return Trade(
                        date=current_date.strftime("%Y-%m-%d"),
                        symbol=nse_sym,
                        direction=direction,
                        sector=sector,
                        entry_time=entry_time_str,
                        entry_price=round(entry_price, 2),
                        stop_loss=round(stop_loss, 2),
                        exit_time=c_ts.strftime("%H:%M"),
                        exit_price=round(exit_price, 2),
                        exit_reason="SQUARE_OFF_1515",
                        pnl_pts=round(pnl_pts, 2),
                        pnl_pct=round(pnl_pct, 2),
                        risk_pts=round(risk_pts, 2),
                        rr_achieved=round(rr, 2),
                    )

        return None

    # â”€â”€ Helpers for slicing multi-index DataFrames â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _get_ticker_day_data(self, ticker: str, date_obj) -> pd.DataFrame:
        """Extracts today's 5m candles for a specific ticker."""
        try:
            if ticker not in self.df_5m["Close"].columns:
                return pd.DataFrame()
            day_mask = self.df_5m.index.date == date_obj
            df_day = self.df_5m.loc[day_mask]
            if df_day.empty:
                return pd.DataFrame()

            return pd.DataFrame({
                "Open": df_day["Open"][ticker],
                "High": df_day["High"][ticker],
                "Low": df_day["Low"][ticker],
                "Close": df_day["Close"][ticker],
                "Volume": df_day["Volume"][ticker],
            }).dropna()
        except Exception:
            return pd.DataFrame()

    def _get_daily_close(self, ticker: str, date_obj) -> Optional[float]:
        try:
            if ticker not in self.df_daily["Close"].columns:
                return None
            row = self.df_daily.loc[self.df_daily.index.date == date_obj]
            if not row.empty:
                val = row["Close"][ticker].iloc[0]
                return float(val) if not np.isnan(val) else None
        except Exception:
            pass
        return None

    def _get_daily_high(self, ticker: str, date_obj) -> Optional[float]:
        try:
            if ticker not in self.df_daily["High"].columns:
                return None
            row = self.df_daily.loc[self.df_daily.index.date == date_obj]
            if not row.empty:
                val = row["High"][ticker].iloc[0]
                return float(val) if not np.isnan(val) else None
        except Exception:
            pass
        return None

    def _get_daily_low(self, ticker: str, date_obj) -> Optional[float]:
        try:
            if ticker not in self.df_daily["Low"].columns:
                return None
            row = self.df_daily.loc[self.df_daily.index.date == date_obj]
            if not row.empty:
                val = row["Low"][ticker].iloc[0]
                return float(val) if not np.isnan(val) else None
        except Exception:
            pass
        return None
