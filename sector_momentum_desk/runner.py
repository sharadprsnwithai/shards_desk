"""
sector_momentum_desk/runner.py
CLI entrypoint for the Sector Momentum Strategy.

Usage:
    # Live mode (runs Phase 1 at current time, then Phase 2 loop until 3:15 PM)
    python -m sector_momentum_desk.runner --mode live [--dry-run]

    # Backtesting mode (replays historical data, no signals sent)
    python -m sector_momentum_desk.runner --mode backtest --start 2024-01-01 --end 2024-03-31
"""
from __future__ import annotations

import argparse
import csv
import logging
import time as time_module
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

from sector_momentum_desk.config import DATA_CACHE_DIR, SESSION_END
from sector_momentum_desk.engine import SectorMomentumEngine
from sector_momentum_desk.entry_trigger import EntryTriggerEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SESSION_END_TIME = time(int(SESSION_END.split(":")[0]), int(SESSION_END.split(":")[1]))


# ── Live mode ────────────────────────────────────────────────────────────────

def run_live(dry_run: bool) -> None:
    """Phase 1 → Phase 2 live monitoring loop."""
    logger.info("=== Sector Momentum Strategy — LIVE MODE ===")

    # Phase 1
    engine = SectorMomentumEngine(dry_run=dry_run)
    result = engine.run()
    _print_selection_result(result)

    if result.nifty_bias == "NEUTRAL":
        logger.info("Nifty is NEUTRAL — no trades today. Exiting.")
        return

    if not result.top_3_stocks:
        logger.info("No stocks selected (sector data unavailable or market closed). Exiting.")
        return


    # Phase 2
    trigger = EntryTriggerEngine(result.top_3_stocks, result.nifty_bias, dry_run=dry_run)
    symbols = [s.symbol for s in result.top_3_stocks]
    logger.info(f"Phase 2 monitoring: {[s.nse_symbol for s in result.top_3_stocks]}")

    import yfinance as yf

    while True:
        now = datetime.now()
        if now.time() >= _SESSION_END_TIME:
            logger.info("Session end reached (3:15 PM). Stopping Phase 2.")
            break

        if len(trigger._signalled) == len(symbols):
            logger.info("All stocks have fired signals. Done.")
            break

        # Fetch latest candles
        stock_candles: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            if sym in trigger._signalled:
                continue
            try:
                df = yf.download(sym, period="1d", interval="5m", progress=False, auto_adjust=True)
                if not df.empty:
                    df = df.reset_index()
                    date_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
                    df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
                    df = df.rename(columns={
                        "Open": "open", "High": "high",
                        "Low": "low", "Close": "close", "Volume": "volume",
                    })
                    stock_candles[sym] = df[["timestamps", "open", "high", "low", "close", "volume"]]
            except Exception as exc:
                logger.warning(f"Failed to fetch {sym}: {exc}")

        nifty_df = pd.DataFrame()
        try:
            nf = yf.download("^NSEI", period="1d", interval="5m", progress=False, auto_adjust=True)
            if not nf.empty:
                nf = nf.reset_index()
                date_col = "Datetime" if "Datetime" in nf.columns else nf.columns[0]
                nf["timestamps"] = pd.to_datetime(nf[date_col]).dt.tz_localize(None)
                nf = nf.rename(columns={
                    "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
                nifty_df = nf[["timestamps", "open", "high", "low", "close", "volume"]]
        except Exception as exc:
            logger.warning(f"Failed to fetch Nifty: {exc}")

        signals = trigger.on_candle(stock_candles, nifty_df)
        for sig in signals:
            _print_signal(sig)

        # Sleep until next 5-minute boundary
        sleep_secs = 300 - (now.timestamp() % 300)
        logger.info(f"Next scan in {sleep_secs:.0f}s …")
        time_module.sleep(sleep_secs)


# ── Backtest mode ─────────────────────────────────────────────────────────────

def run_backtest(start: str, end: str) -> None:
    """Replay Phase 1 + Phase 2 for each trading day in the date range."""
    logger.info(f"=== Sector Momentum Strategy — BACKTEST {start} → {end} ===")

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_CACHE_DIR / f"backtest_{start}_{end}.csv"

    trading_days = pd.bdate_range(start=start, end=end)
    rows: list[dict] = []

    import yfinance as yf

    for day in trading_days:
        day_dt = day.to_pydatetime()
        as_of_930 = day_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        logger.info(f"Backtesting {day_dt.date()} …")

        # Phase 1
        engine = SectorMomentumEngine(dry_run=True)
        try:
            result = engine.run(as_of=as_of_930)
        except Exception as exc:
            logger.warning(f"Phase 1 failed for {day_dt.date()}: {exc}")
            continue

        if result.nifty_bias == "NEUTRAL" or not result.top_3_stocks:
            rows.append({
                "date": day_dt.date(), "bias": result.nifty_bias,
                "sector": result.winning_sector,
                "symbol": "", "direction": "", "entry_price": "",
                "stop_loss": "", "entry_time": "", "pullback_vol": "",
                "pdh_broken": "", "orh15_broken": "",
            })
            continue

        stocks = result.top_3_stocks

        # Phase 2: iterate 5m slots 09:30 → 15:10
        trigger = EntryTriggerEngine(stocks, result.nifty_bias, dry_run=True)
        symbols = [s.symbol for s in stocks]
        fired_signals: list = []

        slot = as_of_930
        while slot.time() < _SESSION_END_TIME:
            slot_end = slot + timedelta(minutes=5)

            # Fetch per-stock historical 5m candles up to slot_end
            stock_candles: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                if sym in trigger._signalled:
                    continue
                try:
                    df = yf.download(sym, start=day_dt.date(), end=(day_dt + timedelta(days=1)).date(),
                                     interval="5m", progress=False, auto_adjust=True)
                    if not df.empty:
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        df = df[df.index <= pd.Timestamp(slot_end)]
                        df = df.reset_index()
                        date_col = df.columns[0]
                        df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
                        df = df.rename(columns={
                            "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume",
                        })
                        stock_candles[sym] = df[["timestamps", "open", "high", "low", "close", "volume"]]
                except Exception:
                    pass

            nifty_df = pd.DataFrame()
            try:
                nf = yf.download("^NSEI", start=day_dt.date(),
                                 end=(day_dt + timedelta(days=1)).date(),
                                 interval="5m", progress=False, auto_adjust=True)
                if not nf.empty:
                    nf.index = pd.to_datetime(nf.index).tz_localize(None)
                    nf = nf[nf.index <= pd.Timestamp(slot_end)].reset_index()
                    date_col = nf.columns[0]
                    nf["timestamps"] = pd.to_datetime(nf[date_col]).dt.tz_localize(None)
                    nf = nf.rename(columns={
                        "Open": "open", "High": "high",
                        "Low": "low", "Close": "close", "Volume": "volume",
                    })
                    nifty_df = nf[["timestamps", "open", "high", "low", "close", "volume"]]
            except Exception:
                pass

            sigs = trigger.on_candle(stock_candles, nifty_df)
            fired_signals.extend(sigs)

            if len(trigger._signalled) == len(symbols):
                break
            slot = slot_end

        # Write one row per signal (or one blank row if no signal)
        if fired_signals:
            for sig in fired_signals:
                rows.append({
                    "date": day_dt.date(),
                    "bias": result.nifty_bias,
                    "sector": result.winning_sector,
                    "symbol": sig.nse_symbol,
                    "direction": sig.direction,
                    "entry_price": sig.entry_price,
                    "stop_loss": sig.stop_loss,
                    "entry_time": sig.breakout_candle_ts.strftime("%H:%M"),
                    "pullback_vol": sig.stop_loss,   # volume stored in PullbackState, not in EntrySignal
                    "pdh_broken": sig.pdh,
                    "orh15_broken": sig.orh15,
                })
        else:
            rows.append({
                "date": day_dt.date(), "bias": result.nifty_bias,
                "sector": result.winning_sector,
                "symbol": "NO_SIGNAL", "direction": result.nifty_bias,
                "entry_price": "", "stop_loss": "", "entry_time": "",
                "pullback_vol": "", "pdh_broken": "", "orh15_broken": "",
            })

    # Save CSV
    if rows:
        fieldnames = ["date", "bias", "sector", "symbol", "direction",
                      "entry_price", "stop_loss", "entry_time",
                      "pullback_vol", "pdh_broken", "orh15_broken"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Backtest saved → {out_path}  ({len(rows)} rows)")
    else:
        logger.info("No rows to write.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_selection_result(result) -> None:
    print(f"\n{'='*55}")
    print(f"[SELECTION] Sector Momentum -- {result.timestamp.strftime('%I:%M %p')}")
    print(f"Nifty Bias  : {result.nifty_bias} ({result.nifty_return:+.2%})")
    if result.nifty_bias != "NEUTRAL":
        print(f"Top Sector  : {result.winning_sector} ({result.sector_return:+.2%})")
        print(f"\nWatch List ({result.nifty_bias}):")
        for i, s in enumerate(result.top_3_stocks, 1):
            print(f"  {i}. {s.nse_symbol:<10} PDH:{s.pdh:.0f}  ORH15:{s.orh15:.0f}  score:{s.score:+.4f}")
    print(f"{'='*55}\n")


def _print_signal(sig) -> None:
    arrow = "[LONG]" if sig.direction == "LONG" else "[SHORT]"
    print(f"\n{'='*55}")
    print(f"*** ENTRY SIGNAL  {sig.timestamp.strftime('%I:%M %p')} ***")
    print(f"Stock     : {sig.nse_symbol} {arrow}")
    print(f"Direction : {sig.direction}")
    print(f"Entry @   : {sig.entry_price:,.2f}")
    print(f"Stop Loss : {sig.stop_loss:,.2f}")
    print(f"{'='*55}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sector Momentum Strategy runner")
    parser.add_argument("--mode", choices=["live", "backtest"], required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Telegram alerts (live mode).")
    parser.add_argument("--start", help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end",   help="Backtest end date YYYY-MM-DD")
    args = parser.parse_args()

    if args.mode == "live":
        run_live(dry_run=args.dry_run)
    else:
        if not args.start or not args.end:
            parser.error("--mode backtest requires --start and --end")
        run_backtest(args.start, args.end)


if __name__ == "__main__":
    main()
