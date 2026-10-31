"""
Unit tests for sector_momentum_desk — all 18 tests, fully mocked (no live API calls).
"""
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import numpy as np

import sector_momentum_desk.config as mom_config


def _candle_frame(rows, tz_naive=True):
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    df.index = df.index.set_names(["Timestamp"])
    return df

def _wide(per_ticker: dict, daily=False):
    """Build a multi-index frame shaped like yfinance.download output (field, ticker)."""
    outer = pd.concat(per_ticker.values(), axis=1, keys=list(per_ticker.keys()))
    outer.columns = outer.columns.swaplevel(0, 1)
    return outer.sort_index(axis=1)

def _make_bt(candles_5m: dict, daily: dict):
    from sector_momentum_desk.backtester import SectorMomentumBacktester
    bt = SectorMomentumBacktester(period="6mo", source="zerodha")
    bt.df_5m = _wide({t: _candle_frame(rows) for t, rows in candles_5m.items()})
    bt.df_daily = _wide({t: _candle_frame(rows) for t, rows in daily.items()})
    bt.df_5m.index = pd.to_datetime(bt.df_5m.index)
    bt.df_daily.index = pd.to_datetime(bt.df_daily.index)
    return bt


# ── Shared intraday tapes for a single test day ─────────────────────────────
D2 = "2026-05-04"
D1 = "2026-05-01"

def _ts(day, hh, mm):
    return pd.Timestamp(f"{day} {hh:02d}:{mm:02d}:00")

MOMENT_INCREMENTS = [(_ts(D2, 9, 15 + m * 5 if m < 9 else 9 + (m - 3) * 5,)) for m in range(0, 1)]


def _day_times(day, n=10):
    return [(_ts(day, 9, 15 + i * 5)) for i in range(3)] + [(_ts(day, 9, 30 + i * 5)) for i in range(n - 3)]

def _nifty_day2_green():
    """3 pre-930 candles biasing BULLISH (+0.24%), then green candles to 10:20."""
    out = []
    for i, (hh, mm) in enumerate([(9, 15), (9, 20), (9, 25)]):
        c = 24900 + 20 * (i + 1)
        out.append(((_ts(D2, hh, mm)), c - 20, c, c - 25, c, 1000))
    for i in range(4):
        c = 24980 + 20 * (i + 1)
        out.append(((_ts(D2, 9, 30 + i * 5)), c - 20, c, c - 25, c, 1000))
    return out

def _bank_day2():
    """Sector ranking tape: +0.5% by 09:25 close."""
    out = []
    for i, (hh, mm) in enumerate([(9, 15), (9, 20), (9, 25)]):
        c = 2000 + 4 * (i + 1)
        out.append(((_ts(D2, hh, mm)), c - 4, c, c - 5, c, 1000))
    return out


def _hdfc_volume_tape_day2():
    """HDFC: 09:15 candle huge volume, 09:20/09:25 tiny -> old sum-weights dominate,
    new last-candle weight makes ICICI win. Prices: ret +1.0% by 09:25."""
    out = []
    for i, v in enumerate([50000, 1, 1]):
        hh, mm = [(9, 15), (9, 20), (9, 25)][i]
        c = 1005 + 2 * (i + 1)   # 1007, 1009, 1011 -> +1.1% vs 1000
        out.append((_ts(D2, hh, mm), c - 2, c, c - 3, c, v))
    return out

def _icici_volume_tape_day2():
    """ICICI: steady volume 100 -> last-candle weight 100; ret +0.8% by 09:25."""
    out = []
    for i, (hh, mm) in enumerate([(9, 15), (9, 20), (9, 25)]):
        c = 1002 + 2 * (i + 1)   # 1004, 1006, 1008 -> +0.8%
        out.append((_ts(D2, hh, mm), c - 2, c, c - 3, c, 100))
    return out


def test_backtest_volume_weight_matches_live():
    """Candidate weight must use the last 5m candle before 09:30 (like live at 09:30),
    not the 15-minute volume sum the old backtest used."""
    candles = {
        "^NSEI": _nifty_day2_green(),
        "Bank": _bank_day2(),
        "HDFCBANK": _hdfc_volume_tape_day2(),
        "ICICIBANK": _icici_volume_tape_day2(),
    }
    daily = {
        "HDFCBANK": [(_ts(D1, 9, 15), 1000, 1000, 990, 1000, 0)],
        "ICICIBANK": [(_ts(D1, 9, 15), 1000, 1000, 990, 1000, 0)],
        "Bank": [(_ts(D1, 9, 15), 2000, 2010, 1990, 2000, 0)],
    }
    bt = _make_bt(candles, daily)
    res = bt._backtest_single_day(pd.Timestamp(D2).date(), pd.Timestamp(D1).date())
    assert res.nifty_bias == "BULLISH"
    assert res.winning_sector == "Bank"
    # Old logic (15m volume sum): HDFCBANK tops; New logic (last candle): ICICIBANK tops
    assert res.top_stocks[0] == "ICICIBANK"


def _hdfc_trail_tape():
    """Day1 flat series (seeds the OLD continuous EMA). Day2 inside the 09:30+ session:
    red pullback 09:30, breakout 09:35, rising closes to 10:15, crossover 10:20."""
    day1 = []
    for hh, mm in [(9, 15), (9, 20), (9, 25), (9, 30), (9, 35), (9, 40), (9, 45), (9, 50), (9, 55)]:
        day1.append((_ts(D1, hh, mm), 100, 100, 100, 100, 1))
    day2 = [
        (_ts(D2, 9, 15), 505, 505, 503, 503, 1),
        (_ts(D2, 9, 20), 504, 506, 503, 506, 10),
        (_ts(D2, 9, 25), 506, 507, 503, 504, 5),
        (_ts(D2, 9, 30), 505, 505, 502, 503, 1),    # red pullback (lowest volume, in-session)
        (_ts(D2, 9, 35), 508, 530, 506, 528, 100),  # breakout: PDH+ORH15 breach
    ]
    c = 536
    for i in range(8):  # 09:40 .. 10:15 rising
        hh, mm = (9, 40 + i * 5) if i < 4 else (10, (i - 4) * 5)
        day2.append((_ts(D2, hh, mm), c - 8, c + 2, c - 10, c, 10))
        c += 8
    day2.append((_ts(D2, 10, 20), 592, 610, 505, 505, 10))  # trail crossover candle
    return day1 + day2


def test_backtest_ema_is_per_day():
    """10-EMA trail must be computed on the trading day's candles only (live parity),
    not over the continuous multi-month series (overnight carry-in)."""
    candles = {
        "^NSEI": _nifty_day2_green() + [((_ts(D1, 9, 15)), 100, 100, 100, 100, 1)],
        "Bank": _bank_day2() + [((_ts(D1, 9, 15)), 100, 100, 100, 100, 1)],
        "HDFCBANK": _hdfc_trail_tape(),
    }
    daily = {
        "HDFCBANK": [(_ts(D1, 9, 15), 500, 500, 480, 500, 0)],
        "Bank": [(_ts(D1, 9, 15), 2000, 2010, 1990, 2000, 0)],
    }
    bt = _make_bt(candles, daily)
    res = bt._backtest_single_day(pd.Timestamp(D2).date(), pd.Timestamp(D1).date())
    assert res.nifty_bias == "BULLISH"
    trades = res.trades
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "LONG"
    # Whole-series EMA (old) stays far below price for the whole rising run ->
    # old code rides to the square-off; per-day EMA exits on the 10:20 crossover.
    assert t.exit_reason == "10_EMA_TRAIL"
    assert t.exit_time == "10:20"


# ─── Task 1: Config + Models ────────────────────────────────────────────────

def test_costs_and_slippage_reduce_pnl():
    """Round-trip costs (bps) + gated-entry slippage must hit the recorded PnL."""
    from sector_momentum_desk.backtester import SectorMomentumBacktester, _apply_costs
    from sector_momentum_desk.config import COST_BPS_ROUNDTRIP, BREAKOUT_SLIPPAGE_PCT
    bt = SectorMomentumBacktester(period="6mo", source="zerodha")
    entry, exit_px = 1000.0, 1010.0
    net = _apply_costs(10.0, entry)
    expected_costs = entry * COST_BPS_ROUNDTRIP / 10_000.0
    expected_slip = entry * BREAKOUT_SLIPPAGE_PCT / 100.0
    assert pytest.approx(10.0 - expected_costs - expected_slip) == net

    # winners and losers both pay: net of a gross -20 stays below -20
    assert _apply_costs(-20.0, 1000.0) < -20.0



def test_config_imports():
    from sector_momentum_desk.config import (
        BIAS_THRESHOLD, SL_BUFFER, NUM_TOP_STOCKS, SESSION_END,
        SECTOR_TICKERS, SECTOR_STOCKS, DATA_CACHE_DIR
    )
    assert BIAS_THRESHOLD == 0.0015
    assert SL_BUFFER == 0.0005
    assert NUM_TOP_STOCKS == 3
    assert len(SECTOR_TICKERS) == 8
    assert "IT" in SECTOR_STOCKS
    assert len(SECTOR_STOCKS["IT"]) >= 5


def test_models_instantiate():
    from sector_momentum_desk.models import (
        StockCandidate, SelectionResult, PullbackState, EntrySignal
    )
    sc = StockCandidate(
        symbol="TCS.NS", nse_symbol="TCS", direction="LONG",
        intraday_return=0.008, oi_weight=0.3, score=0.0104,
        pdh=4210.0, pdl=4100.0, orh15=4195.0, orl15=4110.0
    )
    assert sc.pdh == 4210.0

    pb = PullbackState(
        symbol="TCS.NS", candle_ts=datetime.now(),
        candle_open=4220.0, candle_close=4215.0,
        candle_high=4222.0, candle_low=4213.0,
        candle_volume=12000.0, direction="LONG"
    )
    assert pb.candle_volume == 12000.0

    sr = SelectionResult(
        timestamp=datetime.now(), nifty_bias="BULLISH",
        nifty_return=0.002, winning_sector="IT",
        sector_return=0.008, top_3_stocks=[sc], sector_rankings=[]
    )
    assert sr.nifty_bias == "BULLISH"

    sig = EntrySignal(
        timestamp=datetime.now(), symbol="TCS.NS", nse_symbol="TCS",
        direction="LONG", entry_price=4222.0, stop_loss=4211.0,
        pullback_candle_ts=datetime.now(), breakout_candle_ts=datetime.now(),
        nifty_realigned=True, pdh=4210.0, pdl=4100.0,
        orh15=4195.0, orl15=4110.0, structural_breakout=True
    )
    assert sig.structural_breakout is True


# ─── Task 2: Nifty Bias Detector ────────────────────────────────────────────

def test_bias_bullish():
    from sector_momentum_desk.engine import SectorMomentumEngine
    e = SectorMomentumEngine(dry_run=True)
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15", "2024-01-15 09:20", "2024-01-15 09:25"]),
        "open":  [24000, 24030, 24060],
        "close": [24030, 24060, 24100],
    })
    bias, ret = e._detect_nifty_bias(df)
    assert bias == "BULLISH" and ret > 0.0015


def test_bias_bearish():
    from sector_momentum_desk.engine import SectorMomentumEngine
    e = SectorMomentumEngine(dry_run=True)
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15", "2024-01-15 09:20", "2024-01-15 09:25"]),
        "open":  [24000, 23980, 23950],
        "close": [23980, 23950, 23900],
    })
    bias, ret = e._detect_nifty_bias(df)
    assert bias == "BEARISH" and ret < -0.0015


def test_bias_neutral():
    from sector_momentum_desk.engine import SectorMomentumEngine
    e = SectorMomentumEngine(dry_run=True)
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15", "2024-01-15 09:20", "2024-01-15 09:25"]),
        "open":  [24000, 24001, 24002],
        "close": [24001, 24002, 24003],
    })
    bias, ret = e._detect_nifty_bias(df)
    assert bias == "NEUTRAL"


# ─── Task 3: Sector Ranker ───────────────────────────────────────────────────

def test_sector_ranking_bullish():
    from sector_momentum_desk.sector_ranker import SectorRanker
    from sector_momentum_desk.config import SECTOR_TICKERS

    ranker = SectorRanker()
    mock = {"IT": 0.008, "Bank": 0.005, "Auto": 0.003, "Pharma": 0.001,
            "FMCG": -0.001, "Metal": -0.002, "Realty": -0.004, "Energy": -0.006}
    inv = {v: k for k, v in SECTOR_TICKERS.items()}

    with patch.object(ranker, "_fetch_sector_return",
                      side_effect=lambda t, d: mock[inv[t]]):
        winner, ret, rankings = ranker.rank(datetime.now(), "BULLISH")
    assert winner == "IT"
    assert ret == 0.008
    assert rankings[0]["sector"] == "IT"


def test_sector_ranking_bearish():
    from sector_momentum_desk.sector_ranker import SectorRanker
    from sector_momentum_desk.config import SECTOR_TICKERS

    ranker = SectorRanker()
    mock = {"IT": 0.008, "Bank": 0.005, "Auto": 0.003, "Pharma": 0.001,
            "FMCG": -0.001, "Metal": -0.002, "Realty": -0.004, "Energy": -0.006}
    inv = {v: k for k, v in SECTOR_TICKERS.items()}

    with patch.object(ranker, "_fetch_sector_return",
                      side_effect=lambda t, d: mock[inv[t]]):
        winner, ret, rankings = ranker.rank(datetime.now(), "BEARISH")
    assert winner == "Energy"
    assert ret == -0.006


# ─── Task 4: Stock Scorer ────────────────────────────────────────────────────

def test_stock_scorer_top3():
    from sector_momentum_desk.stock_scorer import StockScorer

    scorer = StockScorer()
    mock_data = {
        "TCS.NS":     (0.010, 1_000_000), "INFY.NS":    (0.008, 800_000),
        "WIPRO.NS":   (0.006, 600_000),   "HCLTECH.NS": (0.004, 400_000),
        "TECHM.NS":   (0.002, 200_000),   "LTIM.NS":    (0.001, 100_000),
    }
    mock_levels = {t: (4210.0, 4100.0, 4195.0, 4110.0) for t in mock_data}
    with patch.object(scorer, "_fetch_stock_data",
                      side_effect=lambda t, d: mock_data.get(t, (0.0, 0))), \
         patch.object(scorer, "_fetch_levels",
                      side_effect=lambda t, d: mock_levels.get(t, (0, 0, 0, 0))):
        result = scorer.score_and_select("IT", "BULLISH", datetime.now())
    assert len(result) == 3
    assert result[0].nse_symbol == "TCS"
    assert all(c.direction == "LONG" for c in result)
    assert result[0].pdh == 4210.0


def test_stock_scorer_short3():
    from sector_momentum_desk.stock_scorer import StockScorer

    scorer = StockScorer()
    mock_data = {
        "TCS.NS":     (-0.010, 1_000_000), "INFY.NS":    (-0.008, 800_000),
        "WIPRO.NS":   (-0.006, 600_000),   "HCLTECH.NS": (-0.004, 400_000),
        "TECHM.NS":   (-0.002, 200_000),   "LTIM.NS":    (-0.001, 100_000),
    }
    mock_levels = {t: (4210.0, 4100.0, 4195.0, 4110.0) for t in mock_data}
    with patch.object(scorer, "_fetch_stock_data",
                      side_effect=lambda t, d: mock_data.get(t, (0.0, 0))), \
         patch.object(scorer, "_fetch_levels",
                      side_effect=lambda t, d: mock_levels.get(t, (0, 0, 0, 0))):
        result = scorer.score_and_select("IT", "BEARISH", datetime.now())
    assert len(result) == 3
    assert result[0].nse_symbol == "TCS"
    assert all(c.direction == "SHORT" for c in result)


def test_oi_weight_in_bounds():
    from sector_momentum_desk.stock_scorer import StockScorer

    scorer = StockScorer()
    mock_data = {t: (0.005, v) for t, v in [
        ("TCS.NS", 1_000_000), ("INFY.NS", 500_000), ("WIPRO.NS", 300_000),
        ("HCLTECH.NS", 200_000), ("TECHM.NS", 100_000), ("LTIM.NS", 50_000),
    ]}
    with patch.object(scorer, "_fetch_stock_data",
                      side_effect=lambda t, d: mock_data.get(t, (0.0, 0))), \
         patch.object(scorer, "_fetch_levels", return_value=(0, 0, 0, 0)):
        result = scorer.score_and_select("IT", "BULLISH", datetime.now())
    assert all(0 < c.oi_weight <= 1.0 for c in result)


# ─── Task 5: Engine Orchestrator ────────────────────────────────────────────

def test_selection_result_shape():
    from sector_momentum_desk.engine import SectorMomentumEngine

    e = SectorMomentumEngine(dry_run=True)
    bullish_df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15", "2024-01-15 09:20", "2024-01-15 09:25"]),
        "open":  [24000, 24030, 24060],
        "close": [24030, 24060, 24100],
    })
    candidates = [
        MagicMock(nse_symbol="TCS",  direction="LONG", score=0.010,
                  pdh=4210.0, pdl=4100.0, orh15=4195.0, orl15=4110.0),
        MagicMock(nse_symbol="INFY", direction="LONG", score=0.008,
                  pdh=1830.0, pdl=1780.0, orh15=1818.0, orl15=1784.0),
        MagicMock(nse_symbol="WIPRO", direction="LONG", score=0.006,
                  pdh=580.0,  pdl=560.0,  orh15=576.0,  orl15=562.0),
    ]
    with patch.object(e._data_provider, "fetch_intraday_5m", return_value=bullish_df), \
         patch.object(e._sector_ranker, "rank",
                      return_value=("IT", 0.008, [{"sector": "IT", "return": 0.008}])), \
         patch.object(e._stock_scorer, "score_and_select", return_value=candidates):
        result = e.run(as_of=datetime(2024, 1, 15, 9, 30))
    assert result.nifty_bias == "BULLISH"
    assert result.winning_sector == "IT"
    assert len(result.top_3_stocks) == 3


def test_neutral_returns_empty():
    from sector_momentum_desk.engine import SectorMomentumEngine

    e = SectorMomentumEngine(dry_run=True)
    flat_df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15", "2024-01-15 09:20", "2024-01-15 09:25"]),
        "open":  [24000, 24000, 24000],
        "close": [24001, 24001, 24001],
    })
    with patch.object(e._data_provider, "fetch_intraday_5m", return_value=flat_df):
        result = e.run(as_of=datetime(2024, 1, 15, 9, 30))
    assert result.nifty_bias == "NEUTRAL"
    assert result.top_3_stocks == []


# ─── Task 6: Entry Trigger Engine ───────────────────────────────────────────

def _make_candidate(direction="LONG", pdh=4200.0, pdl=4100.0,
                    orh15=4195.0, orl15=4110.0):
    from sector_momentum_desk.models import StockCandidate
    return StockCandidate(
        symbol="TCS.NS", nse_symbol="TCS", direction=direction,
        intraday_return=0.01 if direction == "LONG" else -0.01,
        oi_weight=0.3, score=0.013 if direction == "LONG" else -0.013,
        pdh=pdh, pdl=pdl, orh15=orh15, orl15=orl15,
    )


def _bullish_candles():
    """5 candles; 09:40 is RED with session-minimum volume."""
    return pd.DataFrame({
        "timestamps": pd.to_datetime([
            "2024-01-15 09:30", "2024-01-15 09:35",
            "2024-01-15 09:40", "2024-01-15 09:45", "2024-01-15 09:50",
        ]),
        "open":   [4200, 4210, 4215, 4212, 4220],
        "close":  [4210, 4215, 4210, 4220, 4230],   # 09:40: RED (4210<4215)
        "high":   [4215, 4220, 4218, 4225, 4235],
        "low":    [4198, 4208, 4208, 4210, 4218],
        "volume": [50000, 45000, 10000, 40000, 48000],  # 09:40: min
    })


def _bearish_candles():
    """5 candles; 09:40 is GREEN with session-minimum volume."""
    return pd.DataFrame({
        "timestamps": pd.to_datetime([
            "2024-01-15 09:30", "2024-01-15 09:35",
            "2024-01-15 09:40", "2024-01-15 09:45", "2024-01-15 09:50",
        ]),
        "open":   [4200, 4185, 4170, 4175, 4160],
        "close":  [4185, 4170, 4178, 4160, 4145],   # 09:40: GREEN (4178>4170)
        "high":   [4205, 4188, 4180, 4178, 4165],
        "low":    [4180, 4165, 4168, 4158, 4140],
        "volume": [50000, 45000, 10000, 40000, 48000],
    })


def _nifty_green():
    return pd.DataFrame({
        "timestamps": [pd.Timestamp("2024-01-15 10:15")],
        "open": [24050.0], "close": [24075.0],
        "high": [24080.0], "low": [24048.0], "volume": [150000],
    })


def _nifty_red():
    return pd.DataFrame({
        "timestamps": [pd.Timestamp("2024-01-15 10:15")],
        "open": [24075.0], "close": [24050.0],
        "high": [24080.0], "low": [24048.0], "volume": [150000],
    })


def test_pullback_candle_bullish():
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    candidate = _make_candidate("LONG")
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)
    pb = engine._find_pullback("TCS.NS", _bullish_candles(), "BULLISH")
    assert pb is not None
    assert pb.candle_ts == pd.Timestamp("2024-01-15 09:40")
    assert pb.candle_volume == 10000


def test_pullback_candle_bearish():
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    candidate = _make_candidate("SHORT")
    engine = EntryTriggerEngine([candidate], bias="BEARISH", dry_run=True)
    pb = engine._find_pullback("TCS.NS", _bearish_candles(), "BEARISH")
    assert pb is not None
    assert pb.candle_ts == pd.Timestamp("2024-01-15 09:40")


def test_pullback_replaced_by_lower_vol():
    """C_pb updates when an even-lower-volume opposite candle appears later."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    candidate = _make_candidate("LONG")
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)

    # 09:40 RED vol=12000; 09:50 RED vol=8000 (new minimum)
    df = pd.DataFrame({
        "timestamps": pd.to_datetime([
            "2024-01-15 09:30", "2024-01-15 09:35",
            "2024-01-15 09:40", "2024-01-15 09:45", "2024-01-15 09:50",
        ]),
        "open":   [4200, 4210, 4215, 4225, 4230],
        "close":  [4210, 4215, 4210, 4225, 4220],  # 09:40 & 09:50 RED
        "high":   [4215, 4220, 4218, 4228, 4232],
        "low":    [4198, 4208, 4208, 4222, 4218],
        "volume": [50000, 45000, 12000, 40000, 8000],
    })
    pb = engine._find_pullback("TCS.NS", df, "BULLISH")
    assert pb is not None
    assert pb.candle_volume == 8000


def test_breakout_all_gates_pass():
    """All 4 gates pass → EntrySignal emitted with correct entry_price and stop_loss."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import PullbackState

    candidate = _make_candidate("LONG", pdh=4200.0, orh15=4195.0)
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)

    # Align with the actual 09:40 candle in _bullish_candles():
    # open=4215, close=4210 (RED), high=4218, low=4208, volume=10000 (session min)
    pb = PullbackState(
        symbol="TCS.NS", candle_ts=datetime(2024, 1, 15, 9, 40),
        candle_open=4215.0, candle_close=4210.0,
        candle_high=4218.0, candle_low=4208.0,
        candle_volume=10000.0, direction="LONG",
    )
    engine._pullback_states["TCS.NS"] = pb

    # Breakout candle: high=4230 > pb.candle_high=4218; price > PDH=4200 & ORH15=4195
    bo_candle = pd.Series({
        "timestamps": pd.Timestamp("2024-01-15 10:15"),
        "open": 4213.0, "close": 4228.0, "high": 4230.0, "low": 4211.0, "volume": 38000,
    })

    signals = engine.on_candle(
        {"TCS.NS": _bullish_candles()}, _nifty_green(),
        _override_candle={"TCS.NS": bo_candle},
    )
    assert len(signals) == 1
    sig = signals[0]
    assert sig.symbol == "TCS.NS"
    assert sig.entry_price == 4218.0        # High(C_pb) of the 09:40 candle
    assert sig.stop_loss < sig.entry_price  # SL below entry for LONG
    assert sig.nifty_realigned is True
    assert sig.structural_breakout is True


def test_breakout_gate3_fail_nifty():
    """Nifty candle is RED during BULLISH breakout → no signal."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import PullbackState

    candidate = _make_candidate("LONG", pdh=4200.0, orh15=4195.0)
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)

    pb = PullbackState(
        symbol="TCS.NS", candle_ts=datetime(2024, 1, 15, 10, 10),
        candle_open=4218.0, candle_close=4212.0,
        candle_high=4222.0, candle_low=4210.0,
        candle_volume=12000.0, direction="LONG",
    )
    engine._pullback_states["TCS.NS"] = pb

    bo_candle = pd.Series({
        "timestamps": pd.Timestamp("2024-01-15 10:15"),
        "open": 4213.0, "close": 4228.0, "high": 4230.0, "low": 4211.0, "volume": 38000,
    })
    signals = engine.on_candle(
        {"TCS.NS": _bullish_candles()}, _nifty_red(),
        _override_candle={"TCS.NS": bo_candle},
    )
    assert signals == []


def test_breakout_gate4_fail_structural():
    """Price above PDH but below ORH15 → Gate 4 fails → no signal."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import PullbackState

    # ORH15=4250, breakout high=4230 < 4250 → Gate 4 FAIL
    candidate = _make_candidate("LONG", pdh=4200.0, orh15=4250.0)
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)

    pb = PullbackState(
        symbol="TCS.NS", candle_ts=datetime(2024, 1, 15, 10, 10),
        candle_open=4218.0, candle_close=4212.0,
        candle_high=4222.0, candle_low=4210.0,
        candle_volume=12000.0, direction="LONG",
    )
    engine._pullback_states["TCS.NS"] = pb

    bo_candle = pd.Series({
        "timestamps": pd.Timestamp("2024-01-15 10:15"),
        "open": 4213.0, "close": 4228.0, "high": 4230.0, "low": 4211.0, "volume": 38000,
    })
    signals = engine.on_candle(
        {"TCS.NS": _bullish_candles()}, _nifty_green(),
        _override_candle={"TCS.NS": bo_candle},
    )
    assert signals == []


def test_check_trailing_exit_bullish():
    """10 EMA trailing exit triggers when candle closes below 10 EMA (BULLISH/LONG)."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine

    candidate = _make_candidate("LONG")
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)

    # 15 candles rising then last candle drops below 10 EMA
    prices = [100.0 + i for i in range(14)] + [95.0]  # last candle plunges to 95
    df = pd.DataFrame({
        "timestamps": pd.date_range("2024-01-15 09:30", periods=15, freq="5min"),
        "open": prices, "close": prices,
        "high": [p + 1 for p in prices], "low": [p - 1 for p in prices],
        "volume": [1000] * 15,
    })
    triggered, latest_close, latest_ema = engine.check_trailing_exit("TCS.NS", df)
    assert triggered is True
    assert latest_close < latest_ema


def test_check_trailing_exit_bearish():
    """10 EMA trailing exit triggers when candle closes above 10 EMA (BEARISH/SHORT)."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine

    candidate = _make_candidate("SHORT")
    engine = EntryTriggerEngine([candidate], bias="BEARISH", dry_run=True)

    # 15 candles falling then last candle jumps above 10 EMA
    prices = [100.0 - i for i in range(14)] + [105.0]  # last candle surges to 105
    df = pd.DataFrame({
        "timestamps": pd.date_range("2024-01-15 09:30", periods=15, freq="5min"),
        "open": prices, "close": prices,
        "high": [p + 1 for p in prices], "low": [p - 1 for p in prices],
        "volume": [1000] * 15,
    })
    triggered, latest_close, latest_ema = engine.check_trailing_exit("TCS.NS", df)
    assert triggered is True
    assert latest_close > latest_ema

