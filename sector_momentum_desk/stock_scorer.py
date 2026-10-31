"""
sector_momentum_desk/stock_scorer.py
Scores F&O stocks within the winning sector and selects the top-N candidates.
Also fetches PDH / PDL / ORH15 / ORL15 levels needed for Gate 4.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import numpy as np

from sector_momentum_desk.config import NUM_TOP_STOCKS, SECTOR_STOCKS
from sector_momentum_desk.models import StockCandidate

logger = logging.getLogger(__name__)


class StockScorer:
    """
    For a given sector and bias, scores all constituent stocks and returns
    the top NUM_TOP_STOCKS candidates as StockCandidate objects.
    """

    def score_and_select(
        self,
        sector: str,
        bias: str,
        as_of: datetime,
    ) -> list[StockCandidate]:
        """
        Returns up to NUM_TOP_STOCKS StockCandidate objects.
        BULLISH → highest score first.
        BEARISH → lowest (most negative) score first.
        Returns [] if sector not in SECTOR_STOCKS.
        """
        stock_list = SECTOR_STOCKS.get(sector)
        if not stock_list:
            logger.warning(f"[StockScorer] Sector '{sector}' not in SECTOR_STOCKS.")
            return []

        rows: list[dict] = []
        for yf_ticker, nse_symbol in stock_list:
            ret, vol = self._fetch_stock_data(yf_ticker, as_of)
            pdh, pdl, orh15, orl15 = self._fetch_levels(yf_ticker, as_of)
            rows.append({
                "yf_ticker":  yf_ticker,
                "nse_symbol": nse_symbol,
                "ret":        ret,
                "vol":        max(vol, 1),   # guard: never zero
                "pdh":        pdh,
                "pdl":        pdl,
                "orh15":      orh15,
                "orl15":      orl15,
            })

        # Normalise volume proxy weights across all candidates
        total_vol = sum(r["vol"] for r in rows)
        if total_vol == 0:
            volume_weight_fn = lambda r: 1.0 / len(rows)
        else:
            volume_weight_fn = lambda r: r["vol"] / total_vol

        candidates: list[StockCandidate] = []
        for r in rows:
            vw = volume_weight_fn(r)
            score = r["ret"] * (1.0 + vw)
            candidates.append(StockCandidate(
                symbol=r["yf_ticker"],
                nse_symbol=r["nse_symbol"],
                direction="LONG" if bias == "BULLISH" else "SHORT",
                intraday_return=r["ret"],
                volume_weight=vw,
                score=score,
                pdh=r["pdh"],
                pdl=r["pdl"],
                orh15=r["orh15"],
                orl15=r["orl15"],
            ))

        if bias == "BULLISH":
            candidates.sort(key=lambda c: c.score, reverse=True)
        else:  # BEARISH
            candidates.sort(key=lambda c: c.score, reverse=False)

        return candidates[:NUM_TOP_STOCKS]

    # ── Internal data fetchers ────────────────────────────────────────────────

    def _fetch_stock_data(
        self, yf_ticker: str, as_of: datetime
    ) -> tuple[float, float]:
        """
        Returns (intraday_return, latest_volume).
        intraday_return = (latest_5m_close - prev_day_close) / prev_day_close.
        Returns (0.0, 1) on any failure.
        """
        try:
            import yfinance as yf
            df = yf.download(yf_ticker, period="2d", interval="5m",
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                return (0.0, 1)

            df.index = pd.to_datetime(df.index).tz_localize(None)
            today = as_of.date()
            prev_data  = df[df.index.date < today]
            today_data = df[df.index.date == today]

            if prev_data.empty or today_data.empty:
                return (0.0, 1)

            prev_close   = float(prev_data["Close"].iloc[-1])
            latest_close = float(today_data["Close"].iloc[-1])
            latest_vol   = float(today_data["Volume"].iloc[-1])

            if prev_close == 0:
                return (0.0, 1)

            ret = (latest_close - prev_close) / prev_close
            vol = max(latest_vol, 1)
            return (ret, vol)

        except Exception as exc:
            logger.warning(f"[StockScorer] _fetch_stock_data failed for {yf_ticker}: {exc}")
            return (0.0, 1)

    def _fetch_levels(
        self, yf_ticker: str, as_of: datetime
    ) -> tuple[float, float, float, float]:
        """
        Returns (pdh, pdl, orh15, orl15).
        pdh / pdl  : previous trading day's High / Low from daily data.
        orh15      : max(High) of the first three 5-minute candles (9:15–9:30).
        orl15      : min(Low)  of the first three 5-minute candles (9:15–9:30).
        Returns (0.0, 0.0, 0.0, 0.0) on any failure.
        """
        try:
            import yfinance as yf

            # ── PDH / PDL from daily candles ──────────────────────────────
            daily = yf.download(yf_ticker, period="5d", interval="1d",
                                progress=False, auto_adjust=True)
            if daily is None or daily.empty:
                return (0.0, 0.0, 0.0, 0.0)

            daily.index = pd.to_datetime(daily.index).tz_localize(None)
            today = as_of.date()
            prev_daily = daily[daily.index.date < today]
            if prev_daily.empty:
                return (0.0, 0.0, 0.0, 0.0)

            pdh = float(prev_daily["High"].iloc[-1])
            pdl = float(prev_daily["Low"].iloc[-1])

            # ── ORH15 / ORL15 from 5-minute intraday candles ─────────────
            intra = yf.download(yf_ticker, period="2d", interval="5m",
                                progress=False, auto_adjust=True)
            if intra is None or intra.empty:
                return (pdh, pdl, 0.0, 0.0)

            intra.index = pd.to_datetime(intra.index).tz_localize(None)
            today_intra = intra[intra.index.date == today]

            # First three 5m candles: 09:15, 09:20, 09:25
            opening_candles = today_intra[
                (today_intra.index.hour == 9) &
                (today_intra.index.minute < 30)
            ]

            if opening_candles.empty:
                return (pdh, pdl, 0.0, 0.0)

            orh15 = float(opening_candles["High"].max())
            orl15 = float(opening_candles["Low"].min())
            return (pdh, pdl, orh15, orl15)

        except Exception as exc:
            logger.warning(f"[StockScorer] _fetch_levels failed for {yf_ticker}: {exc}")
            return (0.0, 0.0, 0.0, 0.0)
