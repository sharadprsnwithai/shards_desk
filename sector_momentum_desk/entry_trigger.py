"""
sector_momentum_desk/entry_trigger.py
Phase 2: per-stock pullback candle scanner + 4-gate entry confirmation engine.

Called every 5 minutes with the latest candle history for each selected stock
and the current Nifty 5m candles.  Returns any EntrySignal objects that fired
this tick.

Gate summary
────────────
Gate 1  Pullback candle exists   — lowest-volume opposite-direction candle in session
Gate 2  Breakout fired           — subsequent candle's High/Low breaks C_pb extreme
Gate 3  Nifty realigned          — current Nifty 5m candle is in the bias direction
Gate 4  Structural breakout      — price > PDH AND > ORH15 (LONG) / < PDL AND < ORL15 (SHORT)
"""
from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Optional

import pandas as pd

from sector_momentum_desk.config import SL_BUFFER, SESSION_END
from sector_momentum_desk.models import EntrySignal, PullbackState, StockCandidate

logger = logging.getLogger(__name__)

# Parse SESSION_END once
_SESSION_END_TIME: time = time(
    int(SESSION_END.split(":")[0]),
    int(SESSION_END.split(":")[1]),
)


class EntryTriggerEngine:
    """
    Phase 2 engine.  Instantiated once after Phase 1 with the top_3_stocks list.
    on_candle() is called every 5 minutes.
    """

    def __init__(
        self,
        candidates: list[StockCandidate],
        bias: str,
        dry_run: bool = False,
    ) -> None:
        self._candidates: dict[str, StockCandidate] = {
            c.symbol: c for c in candidates
        }
        self._bias = bias
        self._dry_run = dry_run
        # Current best pullback candle per symbol (None = not yet found)
        self._pullback_states: dict[str, Optional[PullbackState]] = {
            sym: None for sym in self._candidates
        }
        # Symbols that have already emitted a signal today
        self._signalled: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def on_candle(
        self,
        stock_candles: dict[str, pd.DataFrame],
        nifty_candles: pd.DataFrame,
        *,
        _override_candle: Optional[dict[str, pd.Series]] = None,
    ) -> list[EntrySignal]:
        """
        Called every 5 minutes.

        Args:
            stock_candles:   {symbol: pd.DataFrame} — full intraday history so far.
            nifty_candles:   pd.DataFrame — Nifty 50 intraday history so far.
            _override_candle: test-injection — if provided, used as the latest candle
                              for the named symbol instead of stock_candles[sym].iloc[-1].

        Returns:
            List of EntrySignal objects fired this tick (usually 0 or 1).
        """
        signals: list[EntrySignal] = []

        for sym, candidate in self._candidates.items():
            if sym in self._signalled:
                continue

            # ── Resolve the latest candle ─────────────────────────────────
            if _override_candle and sym in _override_candle:
                latest_candle = _override_candle[sym]
                candles_df = stock_candles.get(sym)
                if candles_df is None or candles_df.empty:
                    candles_df = _override_candle_to_df(latest_candle)
            else:
                candles_df = stock_candles.get(sym)
                if candles_df is None or candles_df.empty:
                    continue
                latest_candle = candles_df.iloc[-1]

            # ── SESSION_END guard ─────────────────────────────────────────
            candle_ts = pd.Timestamp(latest_candle["timestamps"])
            if candle_ts.time() >= _SESSION_END_TIME:
                continue

            # ── Gate 1: find / update pullback candle ─────────────────────
            # Only replace the current pullback if a lower-volume candidate
            # is found; preserve the existing state otherwise.
            found = self._find_pullback(sym, candles_df, self._bias)
            current = self._pullback_states[sym]
            if found is not None:
                if current is None or found.candle_volume < current.candle_volume:
                    self._pullback_states[sym] = found
            pb = self._pullback_states[sym]
            if pb is None:
                continue

            # ── Gate 2: breakout check ────────────────────────────────────
            if not self._check_breakout(pb, latest_candle, self._bias):
                continue

            # ── Gate 3: Nifty realignment ─────────────────────────────────
            if not self._check_gate3_nifty(nifty_candles):
                logger.debug(f"[EntryTrigger] {sym}: Gate 3 FAIL (Nifty not aligned).")
                continue

            # ── Gate 4: structural breakout ───────────────────────────────
            if not self._check_gate4_structural(latest_candle, candidate, self._bias):
                logger.debug(f"[EntryTrigger] {sym}: Gate 4 FAIL (PDH/PDL/OR not broken).")
                continue

            # ── All gates pass → emit signal ──────────────────────────────
            if self._bias == "BULLISH":
                entry_price = pb.candle_high
                stop_loss   = pb.candle_low * (1.0 - SL_BUFFER)
            else:
                entry_price = pb.candle_low
                stop_loss   = pb.candle_high * (1.0 + SL_BUFFER)

            sig = EntrySignal(
                timestamp=candle_ts.to_pydatetime(),
                symbol=sym,
                nse_symbol=candidate.nse_symbol,
                direction=candidate.direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                pullback_candle_ts=pb.candle_ts,
                breakout_candle_ts=candle_ts.to_pydatetime(),
                nifty_realigned=True,
                pdh=candidate.pdh,
                pdl=candidate.pdl,
                orh15=candidate.orh15,
                orl15=candidate.orl15,
                structural_breakout=True,
            )
            self._signalled.add(sym)
            signals.append(sig)
            logger.info(
                f"[EntryTrigger] 🚨 SIGNAL {sym} {sig.direction} "
                f"entry={sig.entry_price:.2f} sl={sig.stop_loss:.2f}"
            )
            if not self._dry_run:
                self._send_telegram(sig)

        return signals

    # ── Gate implementations ──────────────────────────────────────────────────

    def _find_pullback(
        self,
        symbol: str,
        candles_df: pd.DataFrame,
        bias: str,
    ) -> Optional[PullbackState]:
        """
        Gate 1: find the lowest-volume candle that is in the OPPOSITE direction to bias.

        BULLISH → opposite = RED (close < open)
        BEARISH → opposite = GREEN (close > open)

        On a volume tie between two opposite-direction candles, the earlier (first)
        occurrence wins.

        Returns None if no qualifying candle exists.
        """
        if candles_df is None or candles_df.empty:
            return None

        if bias == "BULLISH":
            opp_mask = candles_df["close"] < candles_df["open"]   # RED
        else:
            opp_mask = candles_df["close"] > candles_df["open"]   # GREEN

        opp_candles = candles_df[opp_mask]
        if opp_candles.empty:
            return None

        # Global session minimum volume (across ALL candles, not just opposite ones)
        session_min_vol = float(candles_df["volume"].min())

        # Among opposite-direction candles, find the one whose volume equals
        # the session minimum.  If none match exactly, find the one with the
        # lowest volume among opposite candles.
        exact_match = opp_candles[opp_candles["volume"] == session_min_vol]
        if not exact_match.empty:
            row = exact_match.iloc[0]
        else:
            row = opp_candles.loc[opp_candles["volume"].idxmin()]

        return PullbackState(
            symbol=symbol,
            candle_ts=pd.Timestamp(row["timestamps"]).to_pydatetime(),
            candle_open=float(row["open"]),
            candle_close=float(row["close"]),
            candle_high=float(row["high"]),
            candle_low=float(row["low"]),
            candle_volume=float(row["volume"]),
            direction=bias,
        )

    def _check_breakout(
        self,
        pb: PullbackState,
        latest_candle: pd.Series,
        bias: str,
    ) -> bool:
        """Gate 2: breakout of the pullback candle's extreme in the bias direction."""
        if bias == "BULLISH":
            return float(latest_candle["high"]) > pb.candle_high
        return float(latest_candle["low"]) < pb.candle_low

    def _check_gate3_nifty(self, nifty_candles: pd.DataFrame) -> bool:
        """Gate 3: Nifty's latest 5m candle must be green (BULLISH) or red (BEARISH)."""
        if nifty_candles is None or nifty_candles.empty:
            return False
        last = nifty_candles.iloc[-1]
        if self._bias == "BULLISH":
            return float(last["close"]) > float(last["open"])
        return float(last["close"]) < float(last["open"])

    def _check_gate4_structural(
        self,
        latest_candle: pd.Series,
        candidate: StockCandidate,
        bias: str,
    ) -> bool:
        """
        Gate 4: structural breakout confirmed.
        BULLISH: candle High > PDH  AND  candle High > ORH15
        BEARISH: candle Low  < PDL  AND  candle Low  < ORL15

        Returns False if PDH/PDL/ORH15/ORL15 are 0.0 (fetch failed — no level, no signal).
        """
        if bias == "BULLISH":
            if candidate.pdh == 0.0 or candidate.orh15 == 0.0:
                logger.debug(f"[EntryTrigger] Gate 4 SKIP: PDH/ORH15 unavailable for {candidate.symbol}")
                return False
            price = float(latest_candle["high"])
            return price > candidate.pdh and price > candidate.orh15
        else:
            if candidate.pdl == 0.0 or candidate.orl15 == 0.0:
                logger.debug(f"[EntryTrigger] Gate 4 SKIP: PDL/ORL15 unavailable for {candidate.symbol}")
                return False
            price = float(latest_candle["low"])
            return price < candidate.pdl and price < candidate.orl15

    # ── Telegram ──────────────────────────────────────────────────────────────

    def _format_telegram_message(self, sig: EntrySignal) -> str:
        arrow = "🟢" if sig.direction == "LONG" else "🔴"
        return (
            f"🚨 ENTRY SIGNAL — {sig.timestamp.strftime('%I:%M %p')}\n\n"
            f"Stock     : {sig.nse_symbol} ({sig.direction})\n"
            f"Direction : {sig.direction} {arrow}\n"
            f"Entry @   : ₹{sig.entry_price:,.2f}  (break of pullback {'high' if sig.direction=='LONG' else 'low'})\n"
            f"Stop Loss : ₹{sig.stop_loss:,.2f}\n\n"
            f"Pullback Candle : {sig.pullback_candle_ts.strftime('%I:%M %p')}\n"
            f"Breakout Candle : {sig.breakout_candle_ts.strftime('%I:%M %p')}\n\n"
            f"✅ Nifty aligned  : {'GREEN' if sig.direction=='LONG' else 'RED'} candle at breakout\n"
            f"✅ Structural break: Price {'>' if sig.direction=='LONG' else '<'} "
            f"PDH ({sig.pdh:.0f}) & {'ORH15' if sig.direction=='LONG' else 'ORL15'} "
            f"({sig.orh15 if sig.direction=='LONG' else sig.orl15:.0f})"
        )

    def _send_telegram(self, sig: EntrySignal) -> None:
        try:
            from telegram_notifier import TelegramNotifier  # type: ignore
            TelegramNotifier().send(self._format_telegram_message(sig))
        except Exception as exc:
            logger.warning(f"[EntryTrigger] Telegram send failed: {exc}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _override_candle_to_df(candle: pd.Series) -> pd.DataFrame:
    """Wraps a single override candle in a one-row DataFrame for _find_pullback."""
    return pd.DataFrame([candle.to_dict()])
