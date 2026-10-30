# Strategy Specification: Sector Momentum Stock Selection & Entry Trigger Strategy (F&O)

## 1. Executive Summary & Intent

The **Sector Momentum Strategy** is a two-phase, market-open, direction-aware engine for NSE F&O stocks.

**Phase 1 — Stock Selection (runs once at 9:30 AM IST):**
1. Determine the **Nifty 50 intraday bias** (BULLISH / BEARISH / NEUTRAL) from the first 15-minute move.
2. Rank the **8 major NSE sector indices** by their 9:15–9:30 AM return.
3. Pick the **top-performing sector** (BULLISH) or **worst-performing sector** (BEARISH).
4. From that sector's F&O constituents, select the **top 3 stocks** by OI-weighted momentum.

**Phase 2 — Entry Trigger (runs every 5 minutes, per selected stock, from 9:30 AM onwards):**
1. On each new 5-minute candle, look for a **lowest-volume pullback candle** — a candle whose direction is **opposite to the bias** and whose volume is the **lowest in the current intraday session so far**.
2. Once found, **wait for a breakout** of that candle's High (BULLISH) or Low (BEARISH) on a subsequent candle.
3. At the moment of breakout, **re-confirm Nifty 50 direction** — the current Nifty candle must still be in the bias direction.
4. **Confirm structural breakout** — price must be trading above PDH AND above the First-15-Minute High (BULLISH), or below PDL AND below the First-15-Minute Low (BEARISH).
5. If all four gates pass → emit a **SIGNAL** for that stock (entry, stop-loss, direction).

This module is **fully decoupled** from all existing engines. It lives in `sector_momentum_desk/`.

---

## 2. Core Feasibility & Design Decisions

### 2.1 Nifty 50 Direction Check (Phase 1)

- **Data source**: 5-minute candles via `DataProvider.fetch_intraday_5m("^NSEI")`.
- **Direction rule** (deterministic):
  - `nifty_return = (close_9:30 - open_9:15) / open_9:15`
  - `BULLISH` if `nifty_return > +BIAS_THRESHOLD` (default **+0.15%**).
  - `BEARISH` if `nifty_return < -BIAS_THRESHOLD` (default **-0.15%**).
  - `NEUTRAL` → **halt, no trades today**.
- Configurable in `sector_momentum_desk/config.py`.

### 2.2 Sector Coverage (8 Indices)

| Sector | yfinance Ticker |
|---|---|
| IT | `^CNXIT` |
| Bank | `^NSEBANK` |
| Auto | `^CNXAUTO` |
| Pharma | `^CNXPHARMA` |
| FMCG | `^CNXFMCG` |
| Metal | `^CNXMETAL` |
| Realty | `^CNXREALTY` |
| Energy | `^CNXENERGY` |

### 2.3 Stock Ranking (Phase 1, within winning sector)

For each candidate stock:
```
score = intraday_return × (1 + oi_weight)
oi_weight = volume_9:30 / sum(volumes of all candidates)
```
BULLISH → top 3 by score (highest). BEARISH → top 3 by score (lowest/most negative).

### 2.4 Phase 2 — Pullback Candle Detection

Scanned every 5 minutes per stock. The **pullback candle** (`C_pb`) is defined as:

```
For BULLISH bias:
  C_pb is a RED candle (Close < Open)  ← opposite to BULLISH
  AND  Volume(C_pb) == min(Volume of all candles from 9:30 AM to now)

For BEARISH bias:
  C_pb is a GREEN candle (Close > Open)  ← opposite to BEARISH
  AND  Volume(C_pb) == min(Volume of all candles from 9:30 AM to now)
```

Only the **single lowest-volume opposite-direction candle** is tracked per stock at any time. If a new candle has even lower volume and is opposite direction, it replaces the current candidate. The pullback candle is reset (cleared) after a signal fires or after 3:15 PM.

### 2.5 Phase 2 — Breakout Trigger

Once `C_pb` is identified, the engine monitors subsequent candles:

```
BULLISH breakout: price of next candle's High > High(C_pb)
BEARISH breakout: price of next candle's Low  < Low(C_pb)
```

A **breakout candle** (`C_bo`) is the first candle that achieves this. The entry trigger price is:
- BULLISH: `entry = High(C_pb)` (buy stop)
- BEARISH: `entry = Low(C_pb)` (sell stop)

### 2.6 Phase 2 — Four-Gate Entry Confirmation

All four gates must pass simultaneously at the moment `C_bo` is identified:

| Gate | BULLISH Condition | BEARISH Condition |
|---|---|---|
| Gate 1: Pullback exists | `C_pb` identified (lowest-vol red candle) | `C_pb` identified (lowest-vol green candle) |
| Gate 2: Breakout fired | `High(C_bo) > High(C_pb)` | `Low(C_bo) < Low(C_pb)` |
| Gate 3: Nifty realignment | Current Nifty 5m candle: `Close > Open` (green) | Current Nifty 5m candle: `Close < Open` (red) |
| Gate 4: Structural breakout | `Price > PDH` AND `Price > ORH₁₅` | `Price < PDL` AND `Price < ORL₁₅` |

Where:
- `PDH` = Previous Day High of the stock
- `PDL` = Previous Day Low of the stock
- `ORH₁₅ = max(High of 9:15, 9:20, 9:25 candles)` — First 15-minute Opening Range High
- `ORL₁₅ = min(Low of 9:15, 9:20, 9:25 candles)` — First 15-minute Opening Range Low

If any gate fails → **no signal**. Continue monitoring the same stock on the next 5-minute candle.

### 2.7 Stop-Loss

- BULLISH entry: `stop_loss = Low(C_pb) × (1 - SL_BUFFER)` where `SL_BUFFER = 0.0005` (0.05%)
- BEARISH entry: `stop_loss = High(C_pb) × (1 + SL_BUFFER)`

### 2.8 Data Sources

| Data Type | Source | Fallback |
|---|---|---|
| Nifty 5m candles | `DataProvider.fetch_intraday_5m("^NSEI")` | yfinance |
| Sector index returns | yfinance direct | synthetic |
| Stock 5m candles | yfinance | synthetic |
| PDH / PDL | yfinance daily (prev candle) | synthetic |
| OI proxy | yfinance intraday volume | raw volume |

No Shoonya. No ML. Fully deterministic.

---

## 3. Strategy Logic — Full Flow

```
09:30 AM IST — PHASE 1 (runs once)
─────────────────────────────────────────────
│
▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Nifty Bias                                      │
│  return = (close_9:30 - open_9:15) / open_9:15          │
│  BULLISH / BEARISH / NEUTRAL → halt if NEUTRAL          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Rank 8 Sector Indices                           │
│  sector_return = (price_9:30 - prev_close) / prev_close │
│  BULLISH → winner = highest sector                      │
│  BEARISH → winner = lowest sector                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Score & Select Top 3 Stocks                     │
│  score = return × (1 + oi_weight)                       │
│  Emit Telegram: bias, sector, top_3 candidates          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼

EVERY 5 MINUTES — PHASE 2 (runs per selected stock, 9:30 AM → 3:15 PM)
─────────────────────────────────────────────
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Pullback Candle Scan                            │
│  For each stock, on each new 5m candle C_i:             │
│                                                         │
│  BULLISH → is C_i RED and Volume(C_i) == session_min?  │
│  BEARISH → is C_i GREEN and Volume(C_i) == session_min? │
│                                                         │
│  YES → mark C_pb = C_i (replace if lower volume found) │
│  NO  → check if existing C_pb's breakout has fired     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Breakout Detection                              │
│  BULLISH: current candle High > High(C_pb)?             │
│  BEARISH: current candle Low  < Low(C_pb)?              │
│                                                         │
│  NO  → continue to next 5m tick                        │
│  YES → proceed to Gate 3 & 4 checks                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: Gate 3 — Nifty Realignment Check               │
│  Fetch current Nifty 5m candle                          │
│  BULLISH: Nifty candle must be GREEN (Close > Open)    │
│  BEARISH: Nifty candle must be RED   (Close < Open)    │
│                                                         │
│  FAIL → reset C_pb, continue monitoring                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 7: Gate 4 — Structural Breakout Check             │
│  compute PDH, PDL (yesterday's daily high/low)          │
│  compute ORH15 = max(High of 9:15, 9:20, 9:25 candles) │
│  compute ORL15 = min(Low  of 9:15, 9:20, 9:25 candles) │
│                                                         │
│  BULLISH: stock_price > PDH AND stock_price > ORH15    │
│  BEARISH: stock_price < PDL AND stock_price < ORL15    │
│                                                         │
│  FAIL → no signal, continue monitoring                 │
└──────────────────────┬──────────────────────────────────┘
                       │  ALL 4 GATES PASS
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 8: Emit EntrySignal                                │
│  symbol, direction, entry_price, stop_loss              │
│  pullback_candle_high/low, breakout_candle_ts           │
│  Telegram: 🚨 ENTRY SIGNAL alert                        │
│  Reset C_pb for this stock (one signal per stock/day)   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Output Contract

### Phase 1 Output

```python
@dataclass
class StockCandidate:
    symbol: str           # "RELIANCE.NS"
    nse_symbol: str       # "RELIANCE"
    direction: str        # "LONG" | "SHORT"
    intraday_return: float
    oi_weight: float
    score: float
    pdh: float            # Previous Day High
    pdl: float            # Previous Day Low
    orh15: float          # First 15-min Opening Range High
    orl15: float          # First 15-min Opening Range Low

@dataclass
class SelectionResult:
    timestamp: datetime
    nifty_bias: str            # "BULLISH" | "BEARISH" | "NEUTRAL"
    nifty_return: float
    winning_sector: str
    sector_return: float
    top_3_stocks: list[StockCandidate]
    sector_rankings: list[dict]
```

### Phase 2 Output

```python
@dataclass
class PullbackState:
    symbol: str
    candle_ts: datetime        # timestamp of C_pb
    candle_open: float
    candle_close: float
    candle_high: float
    candle_low: float
    candle_volume: float
    direction: str             # "LONG" | "SHORT"

@dataclass
class EntrySignal:
    timestamp: datetime        # time of breakout candle
    symbol: str
    nse_symbol: str
    direction: str             # "LONG" | "SHORT"
    entry_price: float         # High(C_pb) for LONG, Low(C_pb) for SHORT
    stop_loss: float
    pullback_candle_ts: datetime
    breakout_candle_ts: datetime
    nifty_realigned: bool      # True (always True when signal fires)
    pdh: float
    pdl: float
    orh15: float
    orl15: float
    structural_breakout: bool  # True (always True when signal fires)
```

---

## 5. Decoupled Package Structure

```
sector_momentum_desk/
├── __init__.py           # Exports SectorMomentumEngine, EntryTriggerEngine
├── config.py             # All constants + sector/stock maps
├── models.py             # StockCandidate, SelectionResult, PullbackState, EntrySignal
├── sector_ranker.py      # Phase 1: ranks 8 sector indices
├── stock_scorer.py       # Phase 1: scores & picks top 3 stocks; fetches PDH/PDL/OR
├── engine.py             # Phase 1 orchestrator: SectorMomentumEngine
├── entry_trigger.py      # Phase 2: per-stock pullback scan + entry gate logic
└── runner.py             # CLI: --mode live | --mode backtest

tests/
└── test_sector_momentum.py    # 18 unit tests (all mocked)
```

---

## 6. Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `BIAS_THRESHOLD` | `0.0015` | Nifty ±0.15% for directional bias |
| `NUM_TOP_STOCKS` | `3` | Stocks selected from winning sector |
| `SL_BUFFER` | `0.0005` | 0.05% buffer beyond pullback candle for stop-loss |
| `CANDLE_INTERVAL` | `"5m"` | Intraday resolution |
| `SESSION_START` | `"09:15"` | Market open |
| `SELECTION_TIME` | `"09:30"` | Phase 1 runs at this time |
| `SESSION_END` | `"15:15"` | Phase 2 stops; no new signals after this |
| `DATA_CACHE_DIR` | `data_cache/sector_momentum/` | Parquet cache |
| `SECTOR_TICKERS` | 8-entry dict | Sector → yfinance ticker |
| `SECTOR_STOCKS` | 8-sector dict | Sector → [(yf_ticker, nse_symbol), ...] |

---

## 7. Telegram Alert Formats

### Phase 1 — Selection Alert (9:30 AM)
```
🎯 Sector Momentum Selection — 09:30 AM

Nifty Bias : BULLISH (+0.23%)
Top Sector : IT (+0.81%)

📌 Watch List (LONG candidates):
1. TCS    PDH: 4210  ORH15: 4195  score: +0.0091
2. INFY   PDH: 1830  ORH15: 1818  score: +0.0078
3. WIPRO  PDH:  580  ORH15:  576  score: +0.0065

Sector Rankings:
  IT      +0.81% ✅  Bank    +0.31%
  Metal   +0.60%     Pharma  +0.12%
  Auto    +0.44%     FMCG    -0.09%
```

### Phase 2 — Entry Signal Alert
```
🚨 ENTRY SIGNAL — 10:15 AM

Stock     : TCS (LONG)
Direction : LONG 🟢
Entry @   : ₹4,212.00  (break of pullback high)
Stop Loss : ₹4,194.50

Pullback Candle : 10:10 AM  Vol: 12,450 (session low)
Breakout Candle : 10:15 AM

✅ Nifty aligned  : GREEN candle at breakout
✅ Structural break: Price > PDH (4210) & ORH15 (4195)
```

---

## 8. Backtesting Mode

`runner.py --mode backtest --start YYYY-MM-DD --end YYYY-MM-DD`:
- Iterates trading days via `pandas.bdate_range`.
- Phase 1: replays historical 5m data at simulated 9:30 AM.
- Phase 2: replays candle-by-candle forward simulation from 9:30 AM to 3:15 PM.
- Outputs CSV: `data_cache/sector_momentum/backtest_YYYY-MM-DD_YYYY-MM-DD.csv`
  - Columns: `date, bias, sector, symbol, direction, entry_price, stop_loss, entry_time, pullback_vol, pdh_broken, orh15_broken`

---

## 9. Constraints & Non-Goals

- **Selection only** — no live order placement.
- **No ML** — fully rule-based and deterministic.
- **No Shoonya** — yfinance + optional Kite DataProvider.
- **Zero coupling** to `daily_engine.py`, `monthly_engine.py`, `vande_bharat_desk/`.
- **One signal per stock per day** — once `EntrySignal` fires for a stock, that stock's Phase 2 monitor stops.
- **NEUTRAL bias** → no Phase 2 runs at all.
- F&O ban list exclusion: `TODO` comment in `stock_scorer.py`.
- Real OI from Shoonya: future enhancement.

---

## 10. Test Coverage Requirements

| # | Test Name | What It Verifies |
|---|---|---|
| 1 | `test_config_imports` | All config constants importable, correct types |
| 2 | `test_models_instantiate` | All 4 dataclasses instantiate without error |
| 3 | `test_bias_bullish` | Nifty return > threshold → BULLISH |
| 4 | `test_bias_bearish` | Nifty return < -threshold → BEARISH |
| 5 | `test_bias_neutral` | Return within band → NEUTRAL |
| 6 | `test_sector_ranking_bullish` | Highest return sector selected |
| 7 | `test_sector_ranking_bearish` | Lowest return sector selected |
| 8 | `test_stock_scorer_top3` | Top 3 LONG stocks returned with direction=LONG |
| 9 | `test_stock_scorer_short3` | Top 3 SHORT stocks returned with direction=SHORT |
| 10 | `test_oi_weight_in_bounds` | OI weights each in (0, 1] for all candidates |
| 11 | `test_selection_result_shape` | run() returns correct SelectionResult fields |
| 12 | `test_neutral_returns_empty` | NEUTRAL bias → top_3_stocks == [] |
| 13 | `test_pullback_candle_bullish` | RED candle with session-min volume → C_pb identified |
| 14 | `test_pullback_candle_bearish` | GREEN candle with session-min volume → C_pb identified |
| 15 | `test_pullback_replaced_by_lower_vol` | C_pb updated when even-lower-volume opposite candle appears |
| 16 | `test_breakout_all_gates_pass` | All 4 gates → EntrySignal emitted with correct fields |
| 17 | `test_breakout_gate3_fail_nifty` | Nifty candle wrong direction → no signal |
| 18 | `test_breakout_gate4_fail_structural` | Price below PDH or ORH15 → no signal |
