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
# shortcut: Telegram-only exits, no order path — wire a ZerodhaOrderExecutor-style
# layer before this desk ever auto-trades.

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

def run_backtest(period: str = "6mo", source: str = "zerodha") -> None:
    """Run historical backtest over the specified period and print full results."""
    from sector_momentum_desk.backtester import SectorMomentumBacktester

    logger.info(f"=== Sector Momentum Strategy — BACKTEST ({period}, source={source}) ===")
    bt = SectorMomentumBacktester(period=period, source=source)
    bt.load_data()
    results = bt.run()

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DATA_CACHE_DIR / f"backtest_report_{period}.csv"

    print("\n" + "=" * 90)
    print(f"| {'DATE':<10} | {'NIFTY BIAS':<10} | {'RET':<7} | {'WINNING SECTOR':<14} | {'TOP 3 STOCKS':<28} | {'TRADES':<6} |")
    print("=" * 90)

    all_trades = []
    csv_rows = []

    for r in results:
        stocks_str = ", ".join(r.top_stocks[:3]) if r.top_stocks else "-"
        print(f"| {r.date:<10} | {r.nifty_bias:<10} | {r.nifty_return*100:>+5.2f}% | {r.winning_sector:<14} | {stocks_str:<28} | {len(r.trades):<6} |")
        for t in r.trades:
            all_trades.append(t)
            csv_rows.append({
                "date": t.date, "symbol": t.symbol, "direction": t.direction,
                "sector": t.sector, "entry_time": t.entry_time, "entry_price": t.entry_price,
                "stop_loss": t.stop_loss, "exit_time": t.exit_time, "exit_price": t.exit_price,
                "exit_reason": t.exit_reason, "pnl_pts": t.pnl_pts, "pnl_pct": t.pnl_pct,
                "risk_pts": t.risk_pts, "rr_achieved": t.rr_achieved,
            })

    print("=" * 90)
    print("\nTRADES EXECUTION DETAIL:")
    print("=" * 120)
    print(f"| {'DATE':<10} | {'SYMBOL':<10} | {'DIR':<5} | {'ENTRY':<6} | {'PRICE':<9} | {'SL':<9} | {'EXIT':<6} | {'EXIT PX':<9} | {'REASON':<16} | {'PNL PTS':<8} | {'PNL %':<7} | {'R:R':<5} |")
    print("=" * 120)

    for t in all_trades:
        print(f"| {t.date:<10} | {t.symbol:<10} | {t.direction:<5} | {t.entry_time:<6} | {t.entry_price:>9.2f} | {t.stop_loss:>9.2f} | {t.exit_time:<6} | {t.exit_price:>9.2f} | {t.exit_reason:<16} | {t.pnl_pts:>+8.2f} | {t.pnl_pct:>+6.2f}% | {t.rr_achieved:>+5.2f} |")
    print("=" * 120)

    # Save to CSV
    if csv_rows:
        df_csv = pd.DataFrame(csv_rows)
        df_csv.to_csv(out_csv, index=False)
        logger.info(f"Saved {len(csv_rows)} trades to {out_csv}")

    # Metrics
    wins = [t for t in all_trades if t.pnl_pct > 0]
    losses = [t for t in all_trades if t.pnl_pct <= 0]
    win_rate = (len(wins) / len(all_trades) * 100.0) if all_trades else 0.0
    total_pnl = sum(t.pnl_pct for t in all_trades)
    gross_profit = sum(t.pnl_pct for t in wins)
    gross_loss = abs(sum(t.pnl_pct for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

    print("\n" + "=" * 55)
    print("SUMMARY PERFORMANCE METRICS:")
    print("=" * 55)
    print(f"  Total Trading Days : {len(results)}")
    print(f"  Days with Bias     : {len([r for r in results if r.nifty_bias != 'NEUTRAL'])}")
    print(f"  Total Trades Taken : {len(all_trades)}")
    print(f"  Winning Trades     : {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losing Trades      : {len(losses)} ({100-win_rate:.1f}%)")
    print(f"  Total Cumulative % : {total_pnl:+.2f}%")
    print(f"  Gross Profit       : {gross_profit:+.2f}%")
    print(f"  Gross Loss         : -{gross_loss:.2f}%")
    print(f"  Profit Factor      : {profit_factor:.2f}")
    print("=" * 55 + "\n")


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
    parser.add_argument("--period", default="6mo",
                        help="Backtest period (e.g. 1mo, 6mo). Default is 6mo.")
    parser.add_argument("--source", choices=["zerodha", "yfinance"], default="zerodha",
                        help="Data source: 'zerodha' (default, official Kite API) or 'yfinance'.")
    args = parser.parse_args()

    if args.mode == "live":
        run_live(dry_run=args.dry_run)
    else:
        run_backtest(period=args.period, source=args.source)


if __name__ == "__main__":
    main()
