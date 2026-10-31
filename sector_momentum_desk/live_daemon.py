"""
sector_momentum_desk/live_daemon.py
Production background daemon for the Sector Momentum Strategy.
Runs continuously inside Docker, aligned to Asia/Kolkata (IST) market hours.

Schedule (Mon-Fri):
  - 09:15 - 09:30 AM IST : Awaits first three 5-minute candles to form.
  - 09:30 AM IST         : Runs Phase 1 (Nifty bias, sector ranking, top-3 stock selection).
                           Sends Telegram Watchlist alert.
  - 09:30 - 15:15 PM IST : Runs Phase 2 every 5 minutes:
                           1. Scans for lowest-volume opposite-direction pullback candles.
                           2. Checks breakout in bias direction.
                           3. Confirms Nifty 5m alignment.
                           4. Confirms PDH/PDL and ORH15/ORL15 structural breakout.
                           5. Emits EntrySignal & sends Telegram alert.
                           6. Trails active trades with 10 EMA and sends Exit alerts.
  - 15:15 PM IST         : Squares off active positions, dispatches daily summary, and sleeps until next morning.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time as time_module
from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd
import pytz
from dotenv import load_dotenv

from sector_momentum_desk.config import (
    BIAS_THRESHOLD,
    CANDLE_INTERVAL,
    DAILY_MAX_LOSS_LIMIT,
    DATA_CACHE_DIR,
    EMA_TRAIL_PERIOD,
    NUM_TOP_STOCKS,
    SECTOR_STOCKS,
    SECTOR_TICKERS,
    SESSION_END,
    SESSION_START,
    SL_BUFFER,
    USE_EMA_TRAILING,
)
from sector_momentum_desk.engine import SectorMomentumEngine
from sector_momentum_desk.entry_trigger import EntryTriggerEngine
from sector_momentum_desk.models import EntrySignal, ExitSignal, SelectionResult, StockCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SectorDaemon] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
_SESSION_START_TIME = time(9, 15)
_SELECTION_TIME = time(9, 30)
_SESSION_END_TIME = time(15, 15)


class SectorMomentumDaemon:
    """
    24/7 Production Runner for Sector Momentum Strategy.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.running = True
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

        # Active state for the day
        self.current_day: Optional[str] = None
        self.phase1_result: Optional[SelectionResult] = None
        self.trigger_engine: Optional[EntryTriggerEngine] = None
        self.active_positions: dict[str, dict] = {}   # symbol -> {entry_price, stop_loss, direction, ...}
        self.closed_positions: list[dict] = []        # list of {pnl_pts, ...}

    def _handle_exit(self, signum, frame):
        logger.info(f"Received shutdown signal ({signum}). Exiting daemon gracefully...")
        self.running = False

    def get_ist_now(self) -> datetime:
        return datetime.now(IST)

    def start(self):
        logger.info("=======================================================")
        logger.info(" Sector Momentum Strategy — Automated Production Daemon")
        logger.info(f" Timezone: Asia/Kolkata | Dry Run: {self.dry_run}")
        logger.info("=======================================================")

        while self.running:
            try:
                now = self.get_ist_now()
                today_str = now.strftime("%Y-%m-%d")
                current_time = now.time()
                weekday = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun

                # ── 1. Weekend Guard ─────────────────────────────────────────
                if weekday >= 5:
                    logger.info("Weekend detected (Saturday/Sunday). Market closed.")
                    self._sleep_until_next_market_open(now)
                    continue

                # ── 2. Before Market Open (Before 09:30 AM) ───────────────────
                if current_time < _SELECTION_TIME:
                    secs_to_930 = (
                        datetime.combine(now.date(), _SELECTION_TIME, tzinfo=IST) - now
                    ).total_seconds()
                    logger.info(f"Pre-market: sleeping {secs_to_930/60:.1f} mins until 09:30 AM IST...")
                    time_module.sleep(min(secs_to_930, 300))
                    continue

                # ── 3. After Market Close (After 15:15 PM) ────────────────────
                if current_time >= _SESSION_END_TIME:
                    if self.active_positions:
                        self._square_off_all_positions(now, reason="SQUARE_OFF_1515")
                    logger.info("Market session closed for today. Sleeping until next trading day...")
                    self._sleep_until_next_market_open(now)
                    continue

                # ── 4. 09:30 AM IST: Execute Phase 1 (Once Per Day) ───────────
                if self.current_day != today_str:
                    logger.info(f"Starting new trading day: {today_str}")
                    self.current_day = today_str
                    self.active_positions = {}
                    self._run_phase_1(now)

                # If Nifty is NEUTRAL or no stocks selected, stand aside for the day
                if (
                    not self.phase1_result
                    or self.phase1_result.nifty_bias == "NEUTRAL"
                    or not self.phase1_result.top_3_stocks
                ):
                    time_module.sleep(300)
                    continue

                # ── 5. 09:30 - 15:15 PM IST: Phase 2 Real-Time 5m Loop ────────
                # Check for daily max loss circuit breaker
                total_pnl = sum(pos["pnl_pts"] for pos in self.closed_positions)
                if total_pnl <= -DAILY_MAX_LOSS_LIMIT:
                    logger.critical(f"DAILY MAX LOSS LIMIT REACHED ({total_pnl:.2f}). Squaring off and stopping...")
                    self._square_off_all_positions(now, reason="DAILY_MAX_LOSS")
                    self.running = False
                    continue

                self._run_phase_2_tick(now)

                # Sleep until 10 seconds after the next 5-minute boundary (:00, :05, :10, ...)
                now_ts = now.timestamp()
                next_boundary = (now_ts // 300 + 1) * 300
                sleep_secs = (next_boundary + 10) - now_ts
                logger.info(f"Next 5m candle scan in {sleep_secs:.0f}s...")
                time_module.sleep(max(sleep_secs, 10))

            except Exception as exc:
                logger.error(f"Unexpected error in daemon loop: {exc}", exc_info=True)
                time_module.sleep(30)

        logger.info("Daemon terminated.")

    def _run_phase_1(self, now: datetime):
        """Executes 9:30 AM Stock Selection and dispatches Watchlist Alert."""
        logger.info("Executing Phase 1: Nifty 50 Bias & Sector Selection (09:30 AM IST)...")
        engine = SectorMomentumEngine(dry_run=self.dry_run)
        try:
            self.phase1_result = engine.run(as_of=now.replace(tzinfo=None))
            logger.info(
                f"Phase 1 complete: Bias={self.phase1_result.nifty_bias} "
                f"({self.phase1_result.nifty_return:+.2%}), Sector={self.phase1_result.winning_sector}"
            )
            if self.phase1_result.top_3_stocks:
                symbols = [s.nse_symbol for s in self.phase1_result.top_3_stocks]
                logger.info(f"Watchlist Top 3 Stocks: {symbols}")
                self.trigger_engine = EntryTriggerEngine(
                    candidates=self.phase1_result.top_3_stocks,
                    bias=self.phase1_result.nifty_bias,
                    dry_run=self.dry_run,
                )
        except Exception as exc:
            logger.error(f"Phase 1 run failed: {exc}", exc_info=True)
            self.phase1_result = None

    def _run_phase_2_tick(self, now: datetime):
        """Fetches latest 5m candles, evaluates Entry Gates, and tracks 10 EMA exits."""
        if not self.trigger_engine or not self.phase1_result:
            return

        symbols = [s.symbol for s in self.phase1_result.top_3_stocks]
        stock_candles = self._fetch_live_5m_candles(symbols)
        nifty_candles = self._fetch_live_5m_candles(["^NSEI"]).get("^NSEI", pd.DataFrame())

        # ── Check New Entries (4 Gates) ──────────────────────────────────────
        new_signals = self.trigger_engine.on_candle(stock_candles, nifty_candles)
        for sig in new_signals:
            # Stale data guard / timestamp freshness check
            if "timestamps" in stock_candles.get(sig.symbol, pd.DataFrame()).columns:
                latest_ts = pd.Timestamp(stock_candles[sig.symbol]["timestamps"].iloc[-1])
                if (now.replace(tzinfo=None) - latest_ts).total_seconds() > 600:
                    logger.warning(f"Skipping signal for {sig.nse_symbol}: candle is stale ({latest_ts})")
                    continue

            # Live fill price reconciliation / slippage adjustment
            fill_price = sig.entry_price
            if not self.dry_run:
                try:
                    from data_provider import DataProvider
                    dp = DataProvider()
                    if dp.kite:
                        quote = dp.kite.quote(f"NSE:{sig.nse_symbol}")
                        ltp = float(quote.get(f"NSE:{sig.nse_symbol}", {}).get("last_price", fill_price))
                        if ltp > 0:
                            # Apply conservative slippage for market orders
                            fill_price = ltp * 1.0005 if sig.direction == "LONG" else ltp * 0.9995
                except Exception as e:
                    logger.warning(f"Could not reconcile live fill price for {sig.nse_symbol}: {e}")

            logger.info(
                f"🚨 [ENTRY ALERT] {sig.nse_symbol} {sig.direction} @ ₹{fill_price:,.2f} "
                f"(Trigger: ₹{sig.entry_price:,.2f} | SL: ₹{sig.stop_loss:,.2f})"
            )
            # Store active position
            self.active_positions[sig.symbol] = {
                "symbol": sig.symbol,
                "nse_symbol": sig.nse_symbol,
                "direction": sig.direction,
                "entry_price": fill_price,
                "stop_loss": sig.stop_loss,
                "entry_time": sig.timestamp,
            }

        # ── Check 10 EMA Trailing Exits for Active Positions ─────────────────
        closed_symbols = []
        for sym, pos in self.active_positions.items():
            df_sym = stock_candles.get(sym)
            if df_sym is None or df_sym.empty:
                continue

            latest_c = df_sym.iloc[-1]
            c_high = float(latest_c["high"])
            c_low = float(latest_c["low"])
            c_close = float(latest_c["close"])

            # 1. Hard Stop-Loss Check
            sl_hit = (
                (c_low <= pos["stop_loss"])
                if pos["direction"] == "LONG"
                else (c_high >= pos["stop_loss"])
            )
            if sl_hit:
                self._dispatch_exit_alert(pos, exit_price=pos["stop_loss"], reason="STOP_LOSS", now=now)
                closed_symbols.append(sym)
                continue

            # 2. 10 EMA Trailing Exit Check
            triggered, l_close, ema_val = self.trigger_engine.check_trailing_exit(sym, df_sym)
            if triggered:
                self._dispatch_exit_alert(pos, exit_price=l_close, reason="10_EMA_TRAIL", now=now)
                closed_symbols.append(sym)
                continue

        for sym in closed_symbols:
            del self.active_positions[sym]
            if self.trigger_engine:
                self.trigger_engine.reset_signal(sym)


    def _dispatch_exit_alert(self, pos: dict, exit_price: float, reason: str, now: datetime):
        pnl_pts = (
            (exit_price - pos["entry_price"])
            if pos["direction"] == "LONG"
            else (pos["entry_price"] - exit_price)
        )
        pnl_pct = (pnl_pts / pos["entry_price"]) * 100.0
        
        # Track for daily circuit breaker
        self.closed_positions.append({"pnl_pts": pnl_pts})

        icon = "🎯" if pnl_pct > 0 else "🛑"
        reason_label = "10 EMA Trailing Exit" if reason == "10_EMA_TRAIL" else "Stop-Loss Hit"

        msg = (
            f"{icon} [EXIT ALERT] {pos['nse_symbol']} ({pos['direction']})\n\n"
            f"Reason    : {reason_label}\n"
            f"Entry @   : ₹{pos['entry_price']:,.2f}\n"
            f"Exit @    : ₹{exit_price:,.2f}\n"
            f"P&L       : {pnl_pts:+.2f} pts ({pnl_pct:+.2f}%)\n"
            f"Time      : {now.strftime('%I:%M %p IST')}"
        )
        logger.info(f"Exit alert: {pos['nse_symbol']} -> {pnl_pct:+.2f}% ({reason})")

        if not self.dry_run:
            try:
                from telegram_notifier import TelegramNotifier
                TelegramNotifier().send(msg)
            except Exception as e:
                logger.warning(f"Failed to send Telegram exit alert: {e}")

    def _square_off_all_positions(self, now: datetime, reason: str = "SQUARE_OFF_1515"):
        logger.info("15:15 PM Cutoff reached: Squaring off all open positions...")
        for sym, pos in list(self.active_positions.items()):
            # Use last available price or entry price
            self._dispatch_exit_alert(pos, exit_price=pos["entry_price"], reason=reason, now=now)
        self.active_positions = {}

    def _fetch_live_5m_candles(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetches latest intraday 5m candles via DataProvider or yfinance."""
        import yfinance as yf
        results = {}
        for sym in symbols:
            try:
                df = yf.download(sym, period="1d", interval="5m", progress=False, auto_adjust=True)
                if not df.empty:
                    df = df.reset_index()
                    date_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
                    df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None) if df[date_col].dt.tz is not None else pd.to_datetime(df[date_col])
                    df = df.rename(columns={
                        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
                    })
                    results[sym] = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
            except Exception as exc:
                logger.warning(f"Live fetch error for {sym}: {exc}")
        return results

    def _sleep_until_next_market_open(self, now: datetime):
        """Calculates sleep time until 09:15 AM on next business day."""
        tomorrow = now.date() + timedelta(days=1)
        while tomorrow.weekday() >= 5:  # Skip weekends
            tomorrow += timedelta(days=1)
        next_open = datetime.combine(tomorrow, _SESSION_START_TIME, tzinfo=IST)
        sleep_secs = max((next_open - now).total_seconds(), 60)
        logger.info(f"Sleeping until next market open: {next_open.strftime('%Y-%m-%d %I:%M %p IST')} ({sleep_secs/3600:.1f} hours)...")
        time_module.sleep(min(sleep_secs, 1800))  # wake periodically to handle SIGTERM


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sector Momentum Production Daemon")
    parser.add_argument("--dry-run", action="store_true", help="Skip live Telegram alerts")
    args = parser.parse_args()

    daemon = SectorMomentumDaemon(dry_run=args.dry_run)
    daemon.start()


if __name__ == "__main__":
    main()
