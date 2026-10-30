"""
sector_momentum_desk/models.py
Dataclasses for Phase 1 (selection) and Phase 2 (entry trigger) output contracts.
"""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StockCandidate:
    """Phase 1 output: a scored stock candidate from the winning sector."""
    symbol: str           # yfinance ticker, e.g. "RELIANCE.NS"
    nse_symbol: str       # NSE trading symbol, e.g. "RELIANCE"
    direction: str        # "LONG" | "SHORT"
    intraday_return: float
    oi_weight: float      # normalised volume proxy weight in [0, 1]
    score: float          # intraday_return * (1 + oi_weight)
    pdh: float            # Previous Day High
    pdl: float            # Previous Day Low
    orh15: float          # First-15-minute Opening Range High
    orl15: float          # First-15-minute Opening Range Low


@dataclass
class SelectionResult:
    """Phase 1 complete output from SectorMomentumEngine.run()."""
    timestamp: datetime
    nifty_bias: str            # "BULLISH" | "BEARISH" | "NEUTRAL"
    nifty_return: float
    winning_sector: str        # e.g. "IT"; empty string when NEUTRAL
    sector_return: float
    top_3_stocks: list         # list[StockCandidate]; empty when NEUTRAL
    sector_rankings: list      # list[{"sector": str, "return": float}], sorted desc


@dataclass
class PullbackState:
    """Phase 2 internal state: the current lowest-volume opposite-direction candle."""
    symbol: str
    candle_ts: datetime
    candle_open: float
    candle_close: float
    candle_high: float
    candle_low: float
    candle_volume: float
    direction: str        # "LONG" | "SHORT" — same as the overall bias


@dataclass
class EntrySignal:
    """Phase 2 output: a confirmed entry trigger that passed all 4 gates."""
    timestamp: datetime          # time of the breakout candle
    symbol: str
    nse_symbol: str
    direction: str               # "LONG" | "SHORT"
    entry_price: float           # High(C_pb) for LONG, Low(C_pb) for SHORT
    stop_loss: float
    pullback_candle_ts: datetime
    breakout_candle_ts: datetime
    nifty_realigned: bool        # always True when signal fires
    pdh: float
    pdl: float
    orh15: float
    orl15: float
    structural_breakout: bool    # always True when signal fires
