# Sector Momentum Stock Selection & Entry Trigger — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully decoupled, deterministic F&O strategy engine that (a) at 9:30 AM selects the top 3 stocks from the strongest/weakest NSE sector, then (b) every 5 minutes per stock watches for the lowest-volume opposite-direction pullback candle, waits for a breakout, re-confirms Nifty direction, and checks PDH/PDL + first-15-min range before emitting an entry signal.

**Architecture:** Single package `sector_momentum_desk/` with 7 focused modules. Phase 1 (selection) and Phase 2 (entry trigger) are split into `engine.py` and `entry_trigger.py` respectively, sharing models and config. `DataProvider` from project root reused for Nifty 5m only. Zero coupling to other strategy packages.

**Tech Stack:** Python 3.11+, yfinance ≥ 0.2.35, pandas ≥ 2.0, numpy ≥ 1.24, python-dotenv, pytest ≥ 8.0, telegram_notifier (project root)

**Spec:** `docs/superpowers/specs/2026-10-30-sector-momentum-stock-selection-design.md`

---

## Global Constraints

- All code lives in `sector_momentum_desk/` — zero imports from/to `daily_engine.py`, `monthly_engine.py`, `vande_bharat_desk/`
- `DataProvider` (project root) may be used for Nifty 5m only
- `telegram_notifier` (project root) imported only inside `_send_telegram()`, wrapped in try/except
- No live order placement — signal generation only
- No ML/LLM — fully rule-based and deterministic
- `BIAS_THRESHOLD = 0.0015`, `SL_BUFFER = 0.0005`, `NUM_TOP_STOCKS = 3` — all in config
- All tests must mock external data — zero live network calls in pytest
- One signal per stock per day — after signal fires, that stock's Phase 2 monitor stops

---

## Review Focus

1. **Pullback candle volume tie** — if two candles have identical minimum volume and opposite direction, the earlier one wins (first occurrence of minimum).
2. **Breakout candle is also a new pullback candidate** — if the breakout candle itself is lower-volume and opposite direction, it must NOT immediately replace C_pb before the signal is evaluated; signal evaluation happens first.
3. **PDH/PDL unavailable** — if yfinance returns no previous-day daily data, Gate 4 must fail gracefully (no signal) rather than raise.
4. **Market opens at PDH/PDL** — stock price already above PDH at 9:15 AM; ORH15 may be above PDH. Both must be independently checked (`price > PDH AND price > ORH15`; both required, not OR).
5. **Phase 2 runs after 3:15 PM** — any candle at or after `SESSION_END` must not produce a signal even if all gates pass.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `sector_momentum_desk/__init__.py` | Create | Exports `SectorMomentumEngine`, `EntryTriggerEngine` |
| `sector_momentum_desk/config.py` | Create | All constants, sector tickers, stock map |
| `sector_momentum_desk/models.py` | Create | `StockCandidate`, `SelectionResult`, `PullbackState`, `EntrySignal` |
| `sector_momentum_desk/sector_ranker.py` | Create | Fetch & rank 8 sector indices |
| `sector_momentum_desk/stock_scorer.py` | Create | Score top 3 stocks; fetch PDH/PDL/OR ranges |
| `sector_momentum_desk/engine.py` | Create | Phase 1 orchestrator |
| `sector_momentum_desk/entry_trigger.py` | Create | Phase 2: pullback scan + 4-gate entry confirmation |
| `sector_momentum_desk/runner.py` | Create | CLI: `--mode live` / `--mode backtest` |
| `tests/test_sector_momentum.py` | Create | 18 unit tests (all mocked) |

---

## Task 1: Package Scaffold + Config + Models

**Files:**
- Create: `sector_momentum_desk/__init__.py`
- Create: `sector_momentum_desk/config.py`
- Create: `sector_momentum_desk/models.py`

**Interfaces:**
- Produces:
  - `BIAS_THRESHOLD: float = 0.0015`
  - `SL_BUFFER: float = 0.0005`
  - `NUM_TOP_STOCKS: int = 3`
  - `SESSION_END: str = "15:15"`
  - `SECTOR_TICKERS: dict[str, str]` — 8 entries
  - `SECTOR_STOCKS: dict[str, list[tuple[str, str]]]` — 8 sectors, ≥5 stocks each
  - `DATA_CACHE_DIR: Path`
  - `StockCandidate(symbol, nse_symbol, direction, intraday_return, oi_weight, score, pdh, pdl, orh15, orl15)` dataclass
  - `SelectionResult(timestamp, nifty_bias, nifty_return, winning_sector, sector_return, top_3_stocks, sector_rankings)` dataclass
  - `PullbackState(symbol, candle_ts, candle_open, candle_close, candle_high, candle_low, candle_volume, direction)` dataclass
  - `EntrySignal(timestamp, symbol, nse_symbol, direction, entry_price, stop_loss, pullback_candle_ts, breakout_candle_ts, nifty_realigned, pdh, pdl, orh15, orl15, structural_breakout)` dataclass

- [ ] **Step 1: Write the failing import tests**

```python
# tests/test_sector_momentum.py
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
    from datetime import datetime
    sc = StockCandidate(symbol="TCS.NS", nse_symbol="TCS", direction="LONG",
                        intraday_return=0.008, oi_weight=0.3, score=0.0104,
                        pdh=4210.0, pdl=4100.0, orh15=4195.0, orl15=4110.0)
    assert sc.pdh == 4210.0
    pb = PullbackState(symbol="TCS.NS", candle_ts=datetime.now(),
                       candle_open=4220.0, candle_close=4215.0,
                       candle_high=4222.0, candle_low=4213.0,
                       candle_volume=12000.0, direction="LONG")
    assert pb.candle_volume == 12000.0
    sig = EntrySignal(timestamp=datetime.now(), symbol="TCS.NS", nse_symbol="TCS",
                      direction="LONG", entry_price=4222.0, stop_loss=4211.0,
                      pullback_candle_ts=datetime.now(), breakout_candle_ts=datetime.now(),
                      nifty_realigned=True, pdh=4210.0, pdl=4100.0,
                      orh15=4195.0, orl15=4110.0, structural_breakout=True)
    assert sig.structural_breakout is True
```

- [ ] **Step 2: Run — expect ModuleNotFoundError**

```bash
cd D:/code/Sharads_monthly_option_selling_desk
python -m pytest tests/test_sector_momentum.py::test_config_imports tests/test_sector_momentum.py::test_models_instantiate -v
```

- [ ] **Step 3: Create `sector_momentum_desk/__init__.py`** — empty with version comment.

- [ ] **Step 4: Create `sector_momentum_desk/config.py`**

```python
SECTOR_TICKERS = {
    "IT": "^CNXIT", "Bank": "^NSEBANK", "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA", "FMCG": "^CNXFMCG", "Metal": "^CNXMETAL",
    "Realty": "^CNXREALTY", "Energy": "^CNXENERGY",
}

SECTOR_STOCKS = {
    "IT":     [("TCS.NS","TCS"),("INFY.NS","INFY"),("WIPRO.NS","WIPRO"),
               ("HCLTECH.NS","HCLTECH"),("TECHM.NS","TECHM"),("LTIM.NS","LTIM")],
    "Bank":   [("HDFCBANK.NS","HDFCBANK"),("ICICIBANK.NS","ICICIBANK"),
               ("KOTAKBANK.NS","KOTAKBANK"),("AXISBANK.NS","AXISBANK"),
               ("SBIN.NS","SBIN"),("BANKBARODA.NS","BANKBARODA")],
    "Auto":   [("MARUTI.NS","MARUTI"),("TATAMOTORS.NS","TATAMOTORS"),
               ("BAJAJ-AUTO.NS","BAJAJ-AUTO"),("HEROMOTOCO.NS","HEROMOTOCO"),
               ("EICHERMOT.NS","EICHERMOT"),("M&M.NS","M&M")],
    "Pharma": [("SUNPHARMA.NS","SUNPHARMA"),("DRREDDY.NS","DRREDDY"),
               ("CIPLA.NS","CIPLA"),("DIVISLAB.NS","DIVISLAB"),
               ("AUROPHARMA.NS","AUROPHARMA"),("BIOCON.NS","BIOCON")],
    "FMCG":   [("HINDUNILVR.NS","HINDUNILVR"),("ITC.NS","ITC"),
               ("NESTLEIND.NS","NESTLEIND"),("BRITANNIA.NS","BRITANNIA"),
               ("DABUR.NS","DABUR"),("GODREJCP.NS","GODREJCP")],
    "Metal":  [("TATASTEEL.NS","TATASTEEL"),("HINDALCO.NS","HINDALCO"),
               ("JSWSTEEL.NS","JSWSTEEL"),("VEDL.NS","VEDL"),
               ("NATIONALUM.NS","NATIONALUM"),("NMDC.NS","NMDC")],
    "Realty": [("DLF.NS","DLF"),("LODHA.NS","LODHA"),
               ("OBEROIRLTY.NS","OBEROIRLTY"),("PRESTIGE.NS","PRESTIGE"),
               ("GODREJPROP.NS","GODREJPROP"),("SOBHA.NS","SOBHA")],
    "Energy": [("RELIANCE.NS","RELIANCE"),("ONGC.NS","ONGC"),
               ("BPCL.NS","BPCL"),("IOC.NS","IOC"),
               ("NTPC.NS","NTPC"),("POWERGRID.NS","POWERGRID")],
}

BIAS_THRESHOLD = 0.0015
SL_BUFFER      = 0.0005
NUM_TOP_STOCKS = 3
CANDLE_INTERVAL = "5m"
SESSION_START  = "09:15"
SELECTION_TIME = "09:30"
SESSION_END    = "15:15"
DATA_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache" / "sector_momentum"
```

- [ ] **Step 5: Create `sector_momentum_desk/models.py`** — four `@dataclass` definitions matching spec §4 exactly. Only stdlib imports.

- [ ] **Step 6: Run — expect PASS**

```bash
python -m pytest tests/test_sector_momentum.py::test_config_imports tests/test_sector_momentum.py::test_models_instantiate -v
```
Expected: 2 PASS.

- [ ] **Step 7: Commit**

```bash
git add sector_momentum_desk/ tests/test_sector_momentum.py
git commit -m "feat(sector-momentum): scaffold, config, all 4 models"
```

---

## Task 2: Nifty Bias Detector

**Files:**
- Create: `sector_momentum_desk/engine.py` (bias detection only)

**Interfaces:**
- Produces: `SectorMomentumEngine._detect_nifty_bias(df_5m: pd.DataFrame) -> tuple[str, float]`
  - `("BULLISH"|"BEARISH"|"NEUTRAL", nifty_return: float)`
  - Uses `df_5m.iloc[0]["open"]` as open_9:15, `df_5m.iloc[-1]["close"]` as close_9:30

- [ ] **Step 1: Write failing tests**

```python
def test_bias_bullish():
    from sector_momentum_desk.engine import SectorMomentumEngine
    import pandas as pd
    e = SectorMomentumEngine(dry_run=True)
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15","2024-01-15 09:20","2024-01-15 09:25"]),
        "open": [24000,24030,24060], "close": [24030,24060,24100],
    })
    bias, ret = e._detect_nifty_bias(df)
    assert bias == "BULLISH" and ret > 0.0015

def test_bias_bearish():
    from sector_momentum_desk.engine import SectorMomentumEngine
    import pandas as pd
    e = SectorMomentumEngine(dry_run=True)
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15","2024-01-15 09:20","2024-01-15 09:25"]),
        "open": [24000,23980,23950], "close": [23980,23950,23900],
    })
    bias, ret = e._detect_nifty_bias(df)
    assert bias == "BEARISH" and ret < -0.0015

def test_bias_neutral():
    from sector_momentum_desk.engine import SectorMomentumEngine
    import pandas as pd
    e = SectorMomentumEngine(dry_run=True)
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15","2024-01-15 09:20","2024-01-15 09:25"]),
        "open": [24000,24001,24002], "close": [24001,24002,24003],
    })
    bias, ret = e._detect_nifty_bias(df)
    assert bias == "NEUTRAL"
```

- [ ] **Step 2: Run — expect ImportError**

```bash
python -m pytest tests/test_sector_momentum.py::test_bias_bullish tests/test_sector_momentum.py::test_bias_bearish tests/test_sector_momentum.py::test_bias_neutral -v
```

- [ ] **Step 3: Implement `SectorMomentumEngine` in `engine.py`**

Constructor: `__init__(self, dry_run: bool = False)` — stores flag, no API connection yet.

`_detect_nifty_bias(self, df_5m: pd.DataFrame) -> tuple[str, float]`:
- Guard: if `df_5m` empty, return `("NEUTRAL", 0.0)`.
- `ret = (df_5m.iloc[-1]["close"] - df_5m.iloc[0]["open"]) / df_5m.iloc[0]["open"]`
- Return `("BULLISH", ret)` / `("BEARISH", ret)` / `("NEUTRAL", ret)` by comparing to `BIAS_THRESHOLD`.

- [ ] **Step 4: Run — expect 3 PASS**

```bash
python -m pytest tests/test_sector_momentum.py::test_bias_bullish tests/test_sector_momentum.py::test_bias_bearish tests/test_sector_momentum.py::test_bias_neutral -v
```

- [ ] **Step 5: Commit**

```bash
git add sector_momentum_desk/engine.py tests/test_sector_momentum.py
git commit -m "feat(sector-momentum): nifty bias detector"
```

---

## Task 3: Sector Ranker

**Files:**
- Create: `sector_momentum_desk/sector_ranker.py`

**Interfaces:**
- Produces: `SectorRanker.rank(as_of: datetime, bias: str) -> tuple[str, float, list[dict]]`
  - `(winning_sector_name, sector_return, rankings_desc)`
  - Internal: `SectorRanker._fetch_sector_return(ticker: str, as_of: datetime) -> float`

- [ ] **Step 1: Write failing tests**

```python
def test_sector_ranking_bullish():
    from sector_momentum_desk.sector_ranker import SectorRanker
    from sector_momentum_desk.config import SECTOR_TICKERS
    from unittest.mock import patch
    from datetime import datetime

    ranker = SectorRanker()
    mock = {"IT":0.008,"Bank":0.005,"Auto":0.003,"Pharma":0.001,
            "FMCG":-0.001,"Metal":-0.002,"Realty":-0.004,"Energy":-0.006}
    inv = {v: k for k, v in SECTOR_TICKERS.items()}
    with patch.object(ranker, "_fetch_sector_return", side_effect=lambda t,d: mock[inv[t]]):
        winner, ret, rankings = ranker.rank(datetime.now(), "BULLISH")
    assert winner == "IT" and ret == 0.008 and rankings[0]["sector"] == "IT"

def test_sector_ranking_bearish():
    from sector_momentum_desk.sector_ranker import SectorRanker
    from sector_momentum_desk.config import SECTOR_TICKERS
    from unittest.mock import patch
    from datetime import datetime

    ranker = SectorRanker()
    mock = {"IT":0.008,"Bank":0.005,"Auto":0.003,"Pharma":0.001,
            "FMCG":-0.001,"Metal":-0.002,"Realty":-0.004,"Energy":-0.006}
    inv = {v: k for k, v in SECTOR_TICKERS.items()}
    with patch.object(ranker, "_fetch_sector_return", side_effect=lambda t,d: mock[inv[t]]):
        winner, ret, rankings = ranker.rank(datetime.now(), "BEARISH")
    assert winner == "Energy" and ret == -0.006
```

- [ ] **Step 2: Run — expect ImportError**

```bash
python -m pytest tests/test_sector_momentum.py::test_sector_ranking_bullish tests/test_sector_momentum.py::test_sector_ranking_bearish -v
```

- [ ] **Step 3: Implement `sector_ranker.py`**

`_fetch_sector_return(ticker, as_of) -> float`: `yf.download(ticker, period="2d", interval="5m")`, prev-day-close vs latest close, return `0.0` on any exception.

`rank(as_of, bias) -> tuple[str, float, list[dict]]`: iterates `SECTOR_TICKERS`, builds `[{"sector": name, "return": val}]`, sorts descending. BULLISH → `[0]`, BEARISH → `[-1]`. All-zero guard returns `("NONE", 0.0, [])`.

- [ ] **Step 4: Run — expect 2 PASS**

```bash
python -m pytest tests/test_sector_momentum.py::test_sector_ranking_bullish tests/test_sector_momentum.py::test_sector_ranking_bearish -v
```

- [ ] **Step 5: Commit**

```bash
git add sector_momentum_desk/sector_ranker.py tests/test_sector_momentum.py
git commit -m "feat(sector-momentum): sector ranker"
```

---

## Task 4: Stock Scorer (with PDH/PDL/OR Ranges)

**Files:**
- Create: `sector_momentum_desk/stock_scorer.py`

**Interfaces:**
- Produces:
  - `StockScorer.score_and_select(sector, bias, as_of) -> list[StockCandidate]`
  - Internal:
    - `StockScorer._fetch_stock_data(yf_ticker, as_of) -> tuple[float, float]` → `(intraday_return, volume)`
    - `StockScorer._fetch_levels(yf_ticker, as_of) -> tuple[float, float, float, float]` → `(pdh, pdl, orh15, orl15)`

- [ ] **Step 1: Write failing tests**

```python
def test_stock_scorer_top3():
    from sector_momentum_desk.stock_scorer import StockScorer
    from unittest.mock import patch
    from datetime import datetime

    scorer = StockScorer()
    mock_data = {"TCS.NS":(0.010,1000000),"INFY.NS":(0.008,800000),
                 "WIPRO.NS":(0.006,600000),"HCLTECH.NS":(0.004,400000),
                 "TECHM.NS":(0.002,200000),"LTIM.NS":(0.001,100000)}
    mock_levels = {t: (4210.0,4100.0,4195.0,4110.0) for t in mock_data}
    with patch.object(scorer, "_fetch_stock_data", side_effect=lambda t,d: mock_data.get(t,(0.0,0))), \
         patch.object(scorer, "_fetch_levels", side_effect=lambda t,d: mock_levels.get(t,(0,0,0,0))):
        result = scorer.score_and_select("IT", "BULLISH", datetime.now())
    assert len(result) == 3
    assert result[0].nse_symbol == "TCS"
    assert all(c.direction == "LONG" for c in result)
    assert result[0].pdh == 4210.0

def test_stock_scorer_short3():
    from sector_momentum_desk.stock_scorer import StockScorer
    from unittest.mock import patch
    from datetime import datetime

    scorer = StockScorer()
    mock_data = {"TCS.NS":(-0.010,1000000),"INFY.NS":(-0.008,800000),
                 "WIPRO.NS":(-0.006,600000),"HCLTECH.NS":(-0.004,400000),
                 "TECHM.NS":(-0.002,200000),"LTIM.NS":(-0.001,100000)}
    mock_levels = {t: (4210.0,4100.0,4195.0,4110.0) for t in mock_data}
    with patch.object(scorer, "_fetch_stock_data", side_effect=lambda t,d: mock_data.get(t,(0.0,0))), \
         patch.object(scorer, "_fetch_levels", side_effect=lambda t,d: mock_levels.get(t,(0,0,0,0))):
        result = scorer.score_and_select("IT", "BEARISH", datetime.now())
    assert len(result) == 3
    assert result[0].nse_symbol == "TCS"
    assert all(c.direction == "SHORT" for c in result)

def test_oi_weight_in_bounds():
    from sector_momentum_desk.stock_scorer import StockScorer
    from unittest.mock import patch
    from datetime import datetime

    scorer = StockScorer()
    mock_data = {t:(0.005,v) for t,v in [("TCS.NS",1000000),("INFY.NS",500000),
                 ("WIPRO.NS",300000),("HCLTECH.NS",200000),("TECHM.NS",100000),("LTIM.NS",50000)]}
    with patch.object(scorer, "_fetch_stock_data", side_effect=lambda t,d: mock_data.get(t,(0.0,0))), \
         patch.object(scorer, "_fetch_levels", return_value=(0,0,0,0)):
        result = scorer.score_and_select("IT", "BULLISH", datetime.now())
    assert all(0 < c.oi_weight <= 1.0 for c in result)
```

- [ ] **Step 2: Run — expect ImportError**

```bash
python -m pytest tests/test_sector_momentum.py::test_stock_scorer_top3 tests/test_sector_momentum.py::test_stock_scorer_short3 tests/test_sector_momentum.py::test_oi_weight_in_bounds -v
```

- [ ] **Step 3: Implement `stock_scorer.py`**

`_fetch_stock_data(yf_ticker, as_of) -> (float, float)`: `yf.download(ticker, period="2d", interval="5m")`, compute `(intraday_return, latest_volume)`, return `(0.0, 1)` on exception.

`_fetch_levels(yf_ticker, as_of) -> (pdh, pdl, orh15, orl15)`:
- Daily data: `yf.download(ticker, period="5d", interval="1d")` → prev day row → `(pdh, pdl)`.
- 5m data: filter candles with time in 09:15–09:30 range → `orh15 = max(high)`, `orl15 = min(low)`.
- Return `(0.0, 0.0, 0.0, 0.0)` on any exception.

`score_and_select(sector, bias, as_of) -> list[StockCandidate]`:
1. Get stocks from `SECTOR_STOCKS[sector]`.
2. Fetch `(ret, vol)` and `(pdh, pdl, orh15, orl15)` per stock.
3. Normalize `oi_weight_i = vol_i / sum(vols)` (guard: if sum==0 use 1/N).
4. `score_i = ret_i * (1 + oi_weight_i)`.
5. BULLISH: sort descending, top `NUM_TOP_STOCKS`. BEARISH: sort ascending, top `NUM_TOP_STOCKS`.
6. Build `StockCandidate` with all fields including pdh/pdl/orh15/orl15.

- [ ] **Step 4: Run — expect 3 PASS**

```bash
python -m pytest tests/test_sector_momentum.py::test_stock_scorer_top3 tests/test_sector_momentum.py::test_stock_scorer_short3 tests/test_sector_momentum.py::test_oi_weight_in_bounds -v
```

- [ ] **Step 5: Commit**

```bash
git add sector_momentum_desk/stock_scorer.py tests/test_sector_momentum.py
git commit -m "feat(sector-momentum): stock scorer with PDH/PDL/OR levels"
```

---

## Task 5: Phase 1 Engine Orchestrator

**Files:**
- Modify: `sector_momentum_desk/engine.py` — add `run()` and `_send_telegram()`
- Modify: `sector_momentum_desk/__init__.py` — export `SectorMomentumEngine`

**Interfaces:**
- Produces: `SectorMomentumEngine.run(as_of: datetime | None = None) -> SelectionResult`

- [ ] **Step 1: Write failing tests**

```python
def test_selection_result_shape():
    from sector_momentum_desk.engine import SectorMomentumEngine
    from unittest.mock import patch, MagicMock
    import pandas as pd
    from datetime import datetime

    e = SectorMomentumEngine(dry_run=True)
    bullish_df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15","2024-01-15 09:20","2024-01-15 09:25"]),
        "open": [24000,24030,24060], "close": [24030,24060,24100],
    })
    candidates = [MagicMock(nse_symbol="TCS",direction="LONG",score=0.01,pdh=4210.0,pdl=4100.0,orh15=4195.0,orl15=4110.0),
                  MagicMock(nse_symbol="INFY",direction="LONG",score=0.008,pdh=1830.0,pdl=1780.0,orh15=1818.0,orl15=1784.0),
                  MagicMock(nse_symbol="WIPRO",direction="LONG",score=0.006,pdh=580.0,pdl=560.0,orh15=576.0,orl15=562.0)]
    with patch.object(e._data_provider, "fetch_intraday_5m", return_value=bullish_df), \
         patch.object(e._sector_ranker, "rank", return_value=("IT", 0.008, [{"sector":"IT","return":0.008}])), \
         patch.object(e._stock_scorer, "score_and_select", return_value=candidates):
        result = e.run(as_of=datetime(2024,1,15,9,30))
    assert result.nifty_bias == "BULLISH"
    assert result.winning_sector == "IT"
    assert len(result.top_3_stocks) == 3

def test_neutral_returns_empty():
    from sector_momentum_desk.engine import SectorMomentumEngine
    from unittest.mock import patch
    import pandas as pd
    from datetime import datetime

    e = SectorMomentumEngine(dry_run=True)
    flat_df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:15","2024-01-15 09:20","2024-01-15 09:25"]),
        "open": [24000,24000,24000], "close": [24001,24001,24001],
    })
    with patch.object(e._data_provider, "fetch_intraday_5m", return_value=flat_df):
        result = e.run(as_of=datetime(2024,1,15,9,30))
    assert result.nifty_bias == "NEUTRAL"
    assert result.top_3_stocks == []
```

- [ ] **Step 2: Run — expect AttributeError (no `run()` yet)**

```bash
python -m pytest tests/test_sector_momentum.py::test_selection_result_shape tests/test_sector_momentum.py::test_neutral_returns_empty -v
```

- [ ] **Step 3: Complete `engine.py`**

`__init__(self, dry_run=False)`: stores flag, instantiates `DataProvider()` → `self._data_provider`, `SectorRanker()` → `self._sector_ranker`, `StockScorer()` → `self._stock_scorer`.

`run(as_of=None) -> SelectionResult`:
1. `as_of = as_of or datetime.now()`
2. `df = self._data_provider.fetch_intraday_5m("^NSEI", days=1)` filtered to candles ≤ `as_of`
3. `bias, ret = self._detect_nifty_bias(df)`
4. NEUTRAL early-return with empty `SelectionResult`
5. `winner, sect_ret, rankings = self._sector_ranker.rank(as_of, bias)`
6. `stocks = self._stock_scorer.score_and_select(winner, bias, as_of)`
7. Build `SelectionResult`, if not `dry_run` call `self._send_telegram(result)` (lazy-import, try/except)

`_format_telegram_message(result) -> str`: per spec §7 Phase 1 format.

- [ ] **Step 4: Update `__init__.py`**

```python
from sector_momentum_desk.engine import SectorMomentumEngine
__all__ = ["SectorMomentumEngine"]
```

- [ ] **Step 5: Run — expect 2 PASS**

```bash
python -m pytest tests/test_sector_momentum.py::test_selection_result_shape tests/test_sector_momentum.py::test_neutral_returns_empty -v
```

- [ ] **Step 6: Commit**

```bash
git add sector_momentum_desk/engine.py sector_momentum_desk/__init__.py tests/test_sector_momentum.py
git commit -m "feat(sector-momentum): phase 1 engine orchestrator"
```

---

## Task 6: Phase 2 — Entry Trigger Engine

**Files:**
- Create: `sector_momentum_desk/entry_trigger.py`
- Modify: `sector_momentum_desk/__init__.py` — add `EntryTriggerEngine` export

**Interfaces:**
- Consumes: `StockCandidate`, `PullbackState`, `EntrySignal` from models; `SL_BUFFER`, `SESSION_END` from config
- Produces:
  - `EntryTriggerEngine(candidates: list[StockCandidate], bias: str, dry_run: bool = False)`
  - `EntryTriggerEngine.on_candle(stock_candles: dict[str, pd.DataFrame], nifty_candles: pd.DataFrame) -> list[EntrySignal]`
    — called every 5 minutes with the latest full intraday candle history for each stock and Nifty.
    — Returns any signals that fired this tick (empty list if none).
  - Internal:
    - `_find_pullback(symbol, candles_df, bias) -> PullbackState | None`
    - `_check_breakout(pb: PullbackState, latest_candle: pd.Series, bias: str) -> bool`
    - `_check_gate3_nifty(nifty_candles: pd.DataFrame) -> bool`
    - `_check_gate4_structural(latest_candle: pd.Series, candidate: StockCandidate, bias: str) -> bool`

- [ ] **Step 1: Write failing tests**

```python
def test_pullback_candle_bullish():
    """RED candle with session-minimum volume is identified as pullback for BULLISH bias."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import StockCandidate
    import pandas as pd
    from datetime import datetime

    candidate = StockCandidate(symbol="TCS.NS", nse_symbol="TCS", direction="LONG",
                               intraday_return=0.01, oi_weight=0.3, score=0.013,
                               pdh=4210.0, pdl=4100.0, orh15=4195.0, orl15=4110.0)
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)

    # 5 candles; candle at 09:40 is RED with lowest volume
    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:30","2024-01-15 09:35",
                                       "2024-01-15 09:40","2024-01-15 09:45","2024-01-15 09:50"]),
        "open":   [4200,4210,4215,4212,4220],
        "close":  [4210,4215,4210,4220,4230],  # 09:40 close<open → RED
        "high":   [4215,4220,4218,4225,4235],
        "low":    [4198,4208,4208,4210,4218],
        "volume": [50000,45000,10000,40000,48000],  # 09:40 is min
    })
    pb = engine._find_pullback("TCS.NS", df, "BULLISH")
    assert pb is not None
    assert pb.candle_ts == pd.Timestamp("2024-01-15 09:40")
    assert pb.candle_volume == 10000

def test_pullback_candle_bearish():
    """GREEN candle with session-minimum volume is identified as pullback for BEARISH bias."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import StockCandidate
    import pandas as pd
    from datetime import datetime

    candidate = StockCandidate(symbol="TCS.NS", nse_symbol="TCS", direction="SHORT",
                               intraday_return=-0.01, oi_weight=0.3, score=-0.013,
                               pdh=4210.0, pdl=4100.0, orh15=4195.0, orl15=4110.0)
    engine = EntryTriggerEngine([candidate], bias="BEARISH", dry_run=True)

    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:30","2024-01-15 09:35",
                                       "2024-01-15 09:40","2024-01-15 09:45","2024-01-15 09:50"]),
        "open":   [4200,4185,4170,4175,4160],
        "close":  [4185,4170,4178,4160,4145],  # 09:40 close>open → GREEN
        "high":   [4205,4188,4180,4178,4165],
        "low":    [4180,4165,4168,4158,4140],
        "volume": [50000,45000,10000,40000,48000],  # 09:40 is min
    })
    pb = engine._find_pullback("TCS.NS", df, "BEARISH")
    assert pb is not None
    assert pb.candle_ts == pd.Timestamp("2024-01-15 09:40")

def test_pullback_replaced_by_lower_vol():
    """C_pb updates when an even-lower-volume opposite candle appears later."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import StockCandidate
    import pandas as pd

    candidate = StockCandidate(symbol="TCS.NS", nse_symbol="TCS", direction="LONG",
                               intraday_return=0.01, oi_weight=0.3, score=0.013,
                               pdh=4210.0, pdl=4100.0, orh15=4195.0, orl15=4110.0)
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)

    df = pd.DataFrame({
        "timestamps": pd.to_datetime(["2024-01-15 09:30","2024-01-15 09:35",
                                       "2024-01-15 09:40","2024-01-15 09:45","2024-01-15 09:50"]),
        "open":   [4200,4210,4215,4225,4230],
        "close":  [4210,4215,4210,4225,4220],  # 09:40 RED (vol 12000), 09:50 RED (vol 8000)
        "high":   [4215,4220,4218,4228,4232],
        "low":    [4198,4208,4208,4222,4218],
        "volume": [50000,45000,12000,40000,8000],
    })
    pb = engine._find_pullback("TCS.NS", df, "BULLISH")
    # 09:50 is RED (close 4220 < open 4230) and has lowest volume 8000
    assert pb is not None
    assert pb.candle_volume == 8000

def test_breakout_all_gates_pass():
    """All 4 gates pass → EntrySignal emitted with correct entry_price and stop_loss."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import StockCandidate, PullbackState
    import pandas as pd
    from datetime import datetime

    candidate = StockCandidate(symbol="TCS.NS", nse_symbol="TCS", direction="LONG",
                               intraday_return=0.01, oi_weight=0.3, score=0.013,
                               pdh=4200.0, pdl=4100.0, orh15=4195.0, orl15=4110.0)
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)

    pb = PullbackState(symbol="TCS.NS", candle_ts=datetime(2024,1,15,10,10),
                       candle_open=4218.0, candle_close=4212.0,
                       candle_high=4222.0, candle_low=4210.0,
                       candle_volume=12000.0, direction="LONG")
    engine._pullback_states["TCS.NS"] = pb

    # Breakout candle: high > pb.candle_high (4222), price > PDH (4200) and > ORH15 (4195)
    bo_candle = pd.Series({
        "timestamps": pd.Timestamp("2024-01-15 10:15"),
        "open": 4213.0, "close": 4228.0, "high": 4230.0, "low": 4211.0, "volume": 38000
    })
    # Nifty: last candle is GREEN
    nifty_df = pd.DataFrame({
        "timestamps": [pd.Timestamp("2024-01-15 10:15")],
        "open": [24050.0], "close": [24075.0], "high": [24080.0], "low": [24048.0], "volume": [150000]
    })

    signals = engine.on_candle({"TCS.NS": None}, nifty_df,
                               _override_candle={"TCS.NS": bo_candle})
    assert len(signals) == 1
    sig = signals[0]
    assert sig.symbol == "TCS.NS"
    assert sig.entry_price == 4222.0         # High(C_pb)
    assert sig.stop_loss < sig.entry_price   # SL below entry for LONG
    assert sig.nifty_realigned is True
    assert sig.structural_breakout is True

def test_breakout_gate3_fail_nifty():
    """Nifty candle is RED during BULLISH breakout → no signal."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import StockCandidate, PullbackState
    import pandas as pd
    from datetime import datetime

    candidate = StockCandidate(symbol="TCS.NS", nse_symbol="TCS", direction="LONG",
                               intraday_return=0.01, oi_weight=0.3, score=0.013,
                               pdh=4200.0, pdl=4100.0, orh15=4195.0, orl15=4110.0)
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)
    pb = PullbackState(symbol="TCS.NS", candle_ts=datetime(2024,1,15,10,10),
                       candle_open=4218.0, candle_close=4212.0,
                       candle_high=4222.0, candle_low=4210.0,
                       candle_volume=12000.0, direction="LONG")
    engine._pullback_states["TCS.NS"] = pb

    bo_candle = pd.Series({
        "timestamps": pd.Timestamp("2024-01-15 10:15"),
        "open": 4213.0, "close": 4228.0, "high": 4230.0, "low": 4211.0, "volume": 38000
    })
    # Nifty RED — Gate 3 fails
    nifty_df = pd.DataFrame({
        "timestamps": [pd.Timestamp("2024-01-15 10:15")],
        "open": [24075.0], "close": [24050.0], "high": [24080.0], "low": [24048.0], "volume": [150000]
    })
    signals = engine.on_candle({"TCS.NS": None}, nifty_df,
                               _override_candle={"TCS.NS": bo_candle})
    assert signals == []

def test_breakout_gate4_fail_structural():
    """Price above PDH but below ORH15 → Gate 4 fails → no signal."""
    from sector_momentum_desk.entry_trigger import EntryTriggerEngine
    from sector_momentum_desk.models import StockCandidate, PullbackState
    import pandas as pd
    from datetime import datetime

    # ORH15 = 4250 — breakout candle high is 4230, so price < ORH15 → Gate 4 fail
    candidate = StockCandidate(symbol="TCS.NS", nse_symbol="TCS", direction="LONG",
                               intraday_return=0.01, oi_weight=0.3, score=0.013,
                               pdh=4200.0, pdl=4100.0, orh15=4250.0, orl15=4110.0)
    engine = EntryTriggerEngine([candidate], bias="BULLISH", dry_run=True)
    pb = PullbackState(symbol="TCS.NS", candle_ts=datetime(2024,1,15,10,10),
                       candle_open=4218.0, candle_close=4212.0,
                       candle_high=4222.0, candle_low=4210.0,
                       candle_volume=12000.0, direction="LONG")
    engine._pullback_states["TCS.NS"] = pb

    bo_candle = pd.Series({
        "timestamps": pd.Timestamp("2024-01-15 10:15"),
        "open": 4213.0, "close": 4228.0, "high": 4230.0, "low": 4211.0, "volume": 38000
    })
    nifty_df = pd.DataFrame({
        "timestamps": [pd.Timestamp("2024-01-15 10:15")],
        "open": [24050.0], "close": [24075.0], "high": [24080.0], "low": [24048.0], "volume": [150000]
    })
    signals = engine.on_candle({"TCS.NS": None}, nifty_df,
                               _override_candle={"TCS.NS": bo_candle})
    assert signals == []
```

- [ ] **Step 2: Run — expect ImportError**

```bash
python -m pytest tests/test_sector_momentum.py::test_pullback_candle_bullish tests/test_sector_momentum.py::test_pullback_candle_bearish tests/test_sector_momentum.py::test_pullback_replaced_by_lower_vol tests/test_sector_momentum.py::test_breakout_all_gates_pass tests/test_sector_momentum.py::test_breakout_gate3_fail_nifty tests/test_sector_momentum.py::test_breakout_gate4_fail_structural -v
```

- [ ] **Step 3: Implement `entry_trigger.py`**

`EntryTriggerEngine.__init__(candidates, bias, dry_run=False)`:
- `self._candidates: dict[str, StockCandidate]` — keyed by `symbol`
- `self._bias: str`
- `self._pullback_states: dict[str, PullbackState | None]` — initialized to `{sym: None for sym in candidates}`
- `self._signalled: set[str]` — symbols that have already fired a signal today (no re-entry)

`_find_pullback(symbol, candles_df, bias) -> PullbackState | None`:
- Filter `candles_df` to rows where candle is opposite direction: BULLISH → `close < open`, BEARISH → `close > open`.
- Among those, find the row with `volume == candles_df["volume"].min()` (global session minimum). On tie, take first occurrence (lowest index).
- Return `PullbackState` built from that row, or `None` if no opposite-direction candle exists.

`_check_breakout(pb, latest_candle, bias) -> bool`:
- BULLISH: `latest_candle["high"] > pb.candle_high`
- BEARISH: `latest_candle["low"] < pb.candle_low`

`_check_gate3_nifty(nifty_candles) -> bool`:
- Take `nifty_candles.iloc[-1]`.
- BULLISH: `close > open`. BEARISH: `close < open`.

`_check_gate4_structural(latest_candle, candidate, bias) -> bool`:
- BULLISH: `latest_candle["high"] > candidate.pdh AND latest_candle["high"] > candidate.orh15`
- BEARISH: `latest_candle["low"] < candidate.pdl AND latest_candle["low"] < candidate.orl15`

`on_candle(stock_candles, nifty_candles, _override_candle=None) -> list[EntrySignal]`:
- `_override_candle` is a test-injection dict `{symbol: pd.Series}` — if provided, use that series as the latest candle for that stock instead of `stock_candles[symbol].iloc[-1]`.
- Loop over `self._candidates` that are not in `self._signalled`.
- Per stock: get latest candle. Skip if candle timestamp `>= SESSION_END` (parse as time string).
- Update `_pullback_states[sym] = self._find_pullback(sym, candles_df, bias)`.
- If no pullback yet: continue.
- If breakout fired (`_check_breakout`): run Gate 3 and Gate 4.
- If both pass: build `EntrySignal`, add symbol to `self._signalled`, append to results list. If not `dry_run`: send Telegram (lazy import, try/except).
- Return results list.

`EntrySignal` fields:
- `entry_price`: BULLISH → `pb.candle_high`, BEARISH → `pb.candle_low`
- `stop_loss`: BULLISH → `pb.candle_low * (1 - SL_BUFFER)`, BEARISH → `pb.candle_high * (1 + SL_BUFFER)`

- [ ] **Step 4: Run — expect 6 PASS**

```bash
python -m pytest tests/test_sector_momentum.py::test_pullback_candle_bullish tests/test_sector_momentum.py::test_pullback_candle_bearish tests/test_sector_momentum.py::test_pullback_replaced_by_lower_vol tests/test_sector_momentum.py::test_breakout_all_gates_pass tests/test_sector_momentum.py::test_breakout_gate3_fail_nifty tests/test_sector_momentum.py::test_breakout_gate4_fail_structural -v
```

- [ ] **Step 5: Update `__init__.py`**

```python
from sector_momentum_desk.engine import SectorMomentumEngine
from sector_momentum_desk.entry_trigger import EntryTriggerEngine
__all__ = ["SectorMomentumEngine", "EntryTriggerEngine"]
```

- [ ] **Step 6: Commit**

```bash
git add sector_momentum_desk/entry_trigger.py sector_momentum_desk/__init__.py tests/test_sector_momentum.py
git commit -m "feat(sector-momentum): phase 2 entry trigger — pullback scan + 4-gate confirmation"
```

---

## Task 7: CLI Runner + Full Test Suite Verification

**Files:**
- Create: `sector_momentum_desk/runner.py`
- Final check: `tests/test_sector_momentum.py` — all 18 tests pass

**Interfaces:**
- Produces: `python -m sector_momentum_desk.runner --mode live [--dry-run]` and `--mode backtest --start YYYY-MM-DD --end YYYY-MM-DD`

- [ ] **Step 1: Run full suite — fix any regressions first**

```bash
python -m pytest tests/test_sector_momentum.py -v
```
Expected: 18 PASS (2 + 3 + 2 + 3 + 2 + 6 from tasks 1–6). Fix any failures before proceeding.

- [ ] **Step 2: Implement `runner.py`**

`argparse` CLI:

**`--mode live [--dry-run]`:**
1. `result = SectorMomentumEngine(dry_run=dry_run).run()` — Phase 1
2. If `result.nifty_bias == "NEUTRAL"`: print and exit.
3. `trigger = EntryTriggerEngine(result.top_3_stocks, result.nifty_bias, dry_run=dry_run)`
4. Loop every 5 minutes (sleep until next 5m boundary using `datetime.now()` aligned to 5m):
   - Fetch latest 5m candles for each stock and Nifty via yfinance.
   - Call `trigger.on_candle(stock_candles, nifty_df)`.
   - Print any signals to stdout.
   - Break if all stocks signalled or time ≥ 15:15.

**`--mode backtest --start YYYY-MM-DD --end YYYY-MM-DD`:**
1. For each day in `pd.bdate_range(start, end)`:
   - Phase 1: `engine.run(as_of=datetime(y,m,d,9,30))` with `dry_run=True`.
   - If NEUTRAL: write row with `bias=NEUTRAL` to output and continue.
   - Phase 2: iterate 5m timestamps from 09:30 to 15:10 for that day. For each timestamp, slice historical data up to that point, call `trigger.on_candle(...)`.
   - Collect first signal per stock per day.
2. Save CSV to `DATA_CACHE_DIR / f"backtest_{start}_{end}.csv"` with columns: `date, bias, sector, symbol, direction, entry_price, stop_loss, entry_time, pullback_vol, pdh_broken, orh15_broken`.

- [ ] **Step 3: Smoke-test live mode**

```bash
python -m sector_momentum_desk.runner --mode live --dry-run
```
Expected: prints Phase 1 selection result, begins Phase 2 monitoring loop (or halts if NEUTRAL / outside market hours). No crash.

- [ ] **Step 4: Final full test suite**

```bash
python -m pytest tests/test_sector_momentum.py -v
```
Expected: all 18 GREEN.

- [ ] **Step 5: Final commit**

```bash
git add sector_momentum_desk/runner.py tests/test_sector_momentum.py
git commit -m "feat(sector-momentum): CLI runner with live loop and backtest replay — strategy complete"
```

---

## Delivery Summary

| Deliverable | Location |
|---|---|
| Spec | `docs/superpowers/specs/2026-10-30-sector-momentum-stock-selection-design.md` |
| Plan | `docs/superpowers/plans/2026-10-30-sector-momentum-stock-selection.md` |
| Phase 1 package | `sector_momentum_desk/` (config, models, sector_ranker, stock_scorer, engine) |
| Phase 2 trigger | `sector_momentum_desk/entry_trigger.py` |
| CLI | `sector_momentum_desk/runner.py` |
| Tests | `tests/test_sector_momentum.py` (18 tests) |
| Cache output | `data_cache/sector_momentum/` |
