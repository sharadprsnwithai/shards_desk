"""
sector_momentum_desk/engine.py
Phase 1 orchestrator: Nifty bias → sector → top-3 stocks → SelectionResult.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from sector_momentum_desk.config import BIAS_THRESHOLD
from sector_momentum_desk.models import SelectionResult

logger = logging.getLogger(__name__)


class SectorMomentumEngine:
    """
    Phase 1 engine.
    Runs once at 9:30 AM: detects Nifty bias, ranks sectors, scores stocks.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run
        # Import lazily to avoid hard failures when DataProvider creds are absent
        try:
            from data_provider import DataProvider
            self._data_provider = DataProvider()
        except Exception as exc:
            logger.warning(f"[SectorMomentumEngine] DataProvider init failed: {exc}")
            self._data_provider = _NullDataProvider()

        from sector_momentum_desk.sector_ranker import SectorRanker
        from sector_momentum_desk.stock_scorer import StockScorer
        self._sector_ranker = SectorRanker()
        self._stock_scorer = StockScorer()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, as_of: Optional[datetime] = None) -> SelectionResult:
        """
        Run Phase 1.  Returns a SelectionResult.
        If Nifty bias is NEUTRAL, top_3_stocks is empty and Phase 2 should not start.
        """
        as_of = as_of or datetime.now()

        df_5m = self._data_provider.fetch_intraday_5m("^NSEI", days=1)
        # Filter to candles up to as_of
        if not df_5m.empty and "timestamps" in df_5m.columns:
            df_5m["timestamps"] = pd.to_datetime(df_5m["timestamps"])
            df_5m = df_5m[df_5m["timestamps"] <= pd.Timestamp(as_of)]

        bias, nifty_ret = self._detect_nifty_bias(df_5m)

        if bias == "NEUTRAL":
            result = SelectionResult(
                timestamp=as_of,
                nifty_bias="NEUTRAL",
                nifty_return=nifty_ret,
                winning_sector="",
                sector_return=0.0,
                top_3_stocks=[],
                sector_rankings=[],
            )
            logger.info("[SectorMomentumEngine] Nifty NEUTRAL — no trade today.")
            return result

        winner, sect_ret, rankings = self._sector_ranker.rank(as_of, bias)
        stocks = self._stock_scorer.score_and_select(winner, bias, as_of)

        result = SelectionResult(
            timestamp=as_of,
            nifty_bias=bias,
            nifty_return=nifty_ret,
            winning_sector=winner,
            sector_return=sect_ret,
            top_3_stocks=stocks,
            sector_rankings=rankings,
        )
        logger.info(
            f"[SectorMomentumEngine] {bias} | sector={winner} | "
            f"stocks={[s.nse_symbol for s in stocks]}"
        )
        if not self._dry_run:
            self._send_telegram(result)
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_nifty_bias(self, df_5m: pd.DataFrame) -> tuple[str, float]:
        """
        Compute Nifty 50 directional bias from the first 15-minute candles.

        Returns:
            ("BULLISH"|"BEARISH"|"NEUTRAL", nifty_return: float)
        """
        if df_5m is None or df_5m.empty:
            return "NEUTRAL", 0.0

        open_price = float(df_5m.iloc[0]["open"])
        close_price = float(df_5m.iloc[-1]["close"])

        if open_price == 0:
            return "NEUTRAL", 0.0

        ret = (close_price - open_price) / open_price

        if ret > BIAS_THRESHOLD:
            return "BULLISH", ret
        if ret < -BIAS_THRESHOLD:
            return "BEARISH", ret
        return "NEUTRAL", ret

    def _format_telegram_message(self, result: SelectionResult) -> str:
        direction_label = "LONG" if result.nifty_bias == "BULLISH" else "SHORT"
        lines = [
            f"🎯 Sector Momentum Selection — {result.timestamp.strftime('%I:%M %p')}",
            "",
            f"Nifty Bias : {result.nifty_bias} ({result.nifty_return:+.2%})",
            f"Top Sector : {result.winning_sector} ({result.sector_return:+.2%})",
            "",
            f"📌 Watch List ({direction_label} candidates):",
        ]
        for i, s in enumerate(result.top_3_stocks, 1):
            lines.append(
                f"{i}. {s.nse_symbol:<8} PDH: {s.pdh:.0f}  "
                f"ORH15: {s.orh15:.0f}  score: {s.score:+.4f}"
            )
        lines.append("")
        lines.append("Sector Rankings:")
        for r in result.sector_rankings:
            marker = " ✅" if r["sector"] == result.winning_sector else ""
            lines.append(f"  {r['sector']:<8} {r['return']:+.2%}{marker}")
        return "\n".join(lines)

    def _send_telegram(self, result: SelectionResult) -> None:
        try:
            from telegram_notifier import TelegramNotifier  # type: ignore
            notifier = TelegramNotifier()
            notifier.send(self._format_telegram_message(result))
        except Exception as exc:
            logger.warning(f"[SectorMomentumEngine] Telegram send failed: {exc}")


class _NullDataProvider:
    """Fallback when DataProvider cannot be initialised (missing creds / offline)."""

    def fetch_intraday_5m(self, symbol: str, days: int = 1) -> pd.DataFrame:
        import yfinance as yf
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval="5m")
            if not df.empty:
                df = df.reset_index()
                date_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
                df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
                df = df.rename(columns={
                    "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })
                return df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
        except Exception as exc:
            logger.warning(f"[_NullDataProvider] yfinance fetch failed for {symbol}: {exc}")
        return pd.DataFrame()
