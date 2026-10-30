"""
sector_momentum_desk/sector_ranker.py
Fetches and ranks the 8 NSE sector indices by their intraday return at a given time,
then returns the winning sector based on the directional bias.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from sector_momentum_desk.config import SECTOR_TICKERS

logger = logging.getLogger(__name__)


class SectorRanker:
    """
    Ranks 8 NSE sector indices by intraday return.

    BULLISH → winning sector  = highest return
    BEARISH → winning sector  = lowest  return
    """

    def rank(
        self, as_of: datetime, bias: str
    ) -> tuple[str, float, list[dict]]:
        """
        Returns:
            (winning_sector_name, winning_sector_return, rankings_desc)
            where rankings_desc is a list[{"sector": str, "return": float}]
            sorted descending by return.
            If all fetches fail, returns ("NONE", 0.0, []).
        """
        entries: list[dict] = []
        for sector_name, ticker in SECTOR_TICKERS.items():
            ret = self._fetch_sector_return(ticker, as_of)
            entries.append({"sector": sector_name, "return": ret})

        # Sort descending by return
        entries.sort(key=lambda x: x["return"], reverse=True)

        # Guard: all zeros → total fetch failure
        if all(e["return"] == 0.0 for e in entries):
            logger.warning("[SectorRanker] All sector returns are 0.0 — data fetch failed.")
            return ("NONE", 0.0, [])

        if bias == "BULLISH":
            winner = entries[0]
        else:  # BEARISH
            winner = entries[-1]

        return winner["sector"], winner["return"], entries

    def _fetch_sector_return(self, ticker: str, as_of: datetime) -> float:
        """
        Fetches 2 days of 5-minute data for the sector index ticker.
        Returns (latest_close - prev_day_close) / prev_day_close.
        Returns 0.0 on any failure.
        """
        try:
            import yfinance as yf
            df = yf.download(ticker, period="2d", interval="5m", progress=False, auto_adjust=True)
            if df is None or df.empty:
                return 0.0

            df.index = pd.to_datetime(df.index).tz_localize(None)

            today = as_of.date()
            prev_day_data = df[df.index.date < today]
            today_data    = df[df.index.date == today]

            if prev_day_data.empty or today_data.empty:
                return 0.0

            prev_close   = float(prev_day_data["Close"].iloc[-1])
            latest_close = float(today_data["Close"].iloc[-1])

            if prev_close == 0:
                return 0.0

            return (latest_close - prev_close) / prev_close

        except Exception as exc:
            logger.warning(f"[SectorRanker] Fetch failed for {ticker}: {exc}")
            return 0.0
