# Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every critical, serious, and minor issue identified in the pre-live audit so the bot can be trusted with real capital.

**Architecture:** Seven self-contained tasks in dependency order — safety guards first (Tasks 1–4 block bad orders and bad data), then data correctness (Tasks 5–6 fix what gets recorded), then defence-in-depth (Task 7 adds a daily kill switch). Each task ends with passing tests and a commit. No task touches production order paths without first covering them with tests.

**Tech Stack:** Python 3.11, PyTorch, KiteConnect 5, pytest, pytz, pandas, `pandas_market_calendars` (new dep, Task 1).

**Spec:** `docs/superpowers/plans/2025-01-31-production-hardening.md` (this file) — derived from the audit in session context.

---

## Global Constraints

- All tests live under `tests/` and are runnable with `pytest tests/ -v` from the project root.
- Never import a live network client inside a test; mock `DataProvider`, `kite`, and `send_telegram_message`.
- Market-hours guard uses IST (`Asia/Kolkata`) timezone via `pytz` — no `datetime.now()` without a timezone argument in any order path.
- `pandas_market_calendars` must be added to `requirements.txt` exactly as `pandas_market_calendars>=4.3.1`.
- Existing passing tests must not regress. Each task's commit must leave the full test suite green.
- Commit messages follow `fix: <what>` or `feat: <what>` convention.

## Review Focus

1. **LSTM/Kronos weights missing at container boot** — model should refuse to produce a signal, not trade on random weights; test that `HybridSignalEngine.generate_signal()` returns `SIDEWAYS_NO_TRADE` when weights file is absent.
2. **Synthetic data reaching order path** — `DataProvider` must raise `RuntimeError` (not return fake rows) when both Kite and yfinance fail in a live context; test the exception propagates to `run_intraday_entry` which then returns `{}` without placing an order.
3. **NSE holiday on a weekday** — scheduler must not call `run_intraday_entry_fn` on a holiday even if `weekday < 5`; test with a mocked known holiday date.
4. **EOD square-off qty mismatch** — `execute_eod_squareoff` must use `leg["quantity"]` that was saved at entry time, not recompute from `self.lots`; test with a position where entry lots ≠ current config lots.
5. **PE/CE ledger symbol bug** — `_record_trade_ledger` must write each leg's own `tradingsymbol`, not inherit a previous leg's value; test that both rows in a strangle position carry distinct symbols.

---

## File Map

| File | Change |
|------|--------|
| `requirements.txt` | Add `pandas_market_calendars>=4.3.1` |
| `market_calendar.py` | **Create** — `is_nse_trading_day(d)` and `get_nse_holidays(year)` |
| `event_filter.py` | Add `should_skip_trading(d) -> bool` that merges holiday + event calendar |
| `scheduler.py` | Wire `should_skip_trading` into weekday entry/exit/monitor guards |
| `data_provider.py` | Replace synthetic fallbacks with `RuntimeError` in live data path; keep synthetics only when `allow_synthetic=True` (default `False` in prod, `True` in tests) |
| `nifty_direction_engine/lstm_predictor.py` | Raise `RuntimeError` when weights file is missing instead of silently using random weights |
| `nifty_direction_engine/hybrid_engine.py` | Catch `RuntimeError` from both predictors → return `SIDEWAYS_NO_TRADE` with reason |
| `monthly_engine.py` | Remove `FallbackPredictor`; on Kronos failure log + skip that asset entirely |
| `zerodha_executor.py` | Fix `execute_eod_squareoff` qty; fix `_record_trade_ledger` symbol bug |
| `daily_engine.py` | Propagate `RuntimeError` from data fetch; add daily max-loss kill switch check pre-entry |
| `config.py` | Add `max_daily_loss_rs: float` field (default `float("inf")`, set via `MAX_DAILY_LOSS_RS` env var) |
| `tests/test_market_calendar.py` | **Create** |
| `tests/test_event_filter.py` | **Create** |
| `tests/test_data_provider.py` | Extend with synthetic-disabled + RuntimeError cases |
| `tests/test_lstm_predictor.py` | **Create** |
| `tests/test_hybrid_engine.py` | **Create** |
| `tests/test_zerodha_executor.py` | Extend: qty fix, symbol fix, market-hours mock |
| `tests/test_scheduler.py` | Extend: holiday guard test |
| `tests/test_daily_engine.py` | Extend: kill-switch test |

---

## Task 1: NSE Holiday Calendar

**Files:**
- Create: `market_calendar.py`
- Modify: `requirements.txt`
- Test: `tests/test_market_calendar.py`

**Interfaces:**
- Produces:
  - `is_nse_trading_day(d: date) -> bool` — returns `False` on weekends and NSE holidays
  - `get_nse_holidays(year: int) -> list[date]` — returns sorted list of NSE exchange holidays for that year

- [ ] **Step 1: Add dependency**

In `requirements.txt`, add on a new line:
```
pandas_market_calendars>=4.3.1
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_market_calendar.py
from datetime import date
from market_calendar import is_nse_trading_day, get_nse_holidays

def test_weekend_is_not_trading_day():
    assert is_nse_trading_day(date(2025, 1, 4)) is False   # Saturday
    assert is_nse_trading_day(date(2025, 1, 5)) is False   # Sunday

def test_known_nse_holiday_is_not_trading_day():
    # Republic Day 2025 — NSE closed
    assert is_nse_trading_day(date(2025, 1, 26)) is False

def test_normal_weekday_is_trading_day():
    assert is_nse_trading_day(date(2025, 1, 27)) is True   # Monday after Republic Day

def test_get_nse_holidays_returns_list_of_dates():
    holidays = get_nse_holidays(2025)
    assert isinstance(holidays, list)
    assert all(isinstance(h, date) for h in holidays)
    assert date(2025, 1, 26) in holidays   # Republic Day
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd D:/code/Sharads_monthly_option_selling_desk
python -m pytest tests/test_market_calendar.py -v
```
Expected: `ModuleNotFoundError: No module named 'market_calendar'`

- [ ] **Step 4: Implement `market_calendar.py`**

Use `pandas_market_calendars.get_calendar("NSE")` to fetch the schedule for a given year. `get_nse_holidays(year)` returns the list of holiday dates from the calendar's `holidays()` method. `is_nse_trading_day(d)` returns `False` if `d.weekday() >= 5` or `d` is in `get_nse_holidays(d.year)`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_market_calendar.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt market_calendar.py tests/test_market_calendar.py
git commit -m "feat: add NSE trading calendar with holiday guard"
```

---

## Task 2: Wire Holiday Guard into Scheduler and Event Filter

**Files:**
- Modify: `event_filter.py`
- Modify: `scheduler.py`
- Create: `tests/test_event_filter.py`
- Modify: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `is_nse_trading_day(d: date) -> bool` from Task 1
- Produces:
  - `should_skip_trading(d: date) -> bool` in `event_filter.py` — `True` when the day is a holiday OR a high-impact event day

- [ ] **Step 1: Write failing tests**

```python
# tests/test_event_filter.py
from datetime import date
from unittest.mock import patch
from event_filter import should_skip_trading

def test_skip_on_nse_holiday():
    with patch("event_filter.is_nse_trading_day", return_value=False):
        assert should_skip_trading(date(2025, 1, 26)) is True

def test_skip_on_high_impact_event():
    with patch("event_filter.is_nse_trading_day", return_value=True):
        # RBI MPC day in the calendar
        assert should_skip_trading(date(2025, 2, 7)) is True

def test_normal_day_not_skipped():
    with patch("event_filter.is_nse_trading_day", return_value=True):
        assert should_skip_trading(date(2025, 1, 27)) is False
```

```python
# Add to tests/test_scheduler.py
def test_scheduler_skips_entry_on_nse_holiday():
    from unittest.mock import MagicMock, patch
    import pytz
    from datetime import datetime
    from scheduler import check_scheduled_triggers

    tz = pytz.timezone("Asia/Kolkata")
    state = {"last_entry_date": None, "last_exit_date": None,
             "last_monthly_train_date": None, "last_monthly_run_date": None}
    mock_entry = MagicMock()

    # Republic Day 2025 at 10:00 AM — Monday weekday but NSE holiday
    t = tz.localize(datetime(2025, 1, 26, 10, 0, 0))
    with patch("scheduler.should_skip_trading", return_value=True):
        check_scheduled_triggers(t, state, run_intraday_entry_fn=mock_entry,
                                 run_intraday_exit_fn=MagicMock(),
                                 monitor_intraday_fn=None)
    mock_entry.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_event_filter.py tests/test_scheduler.py::test_scheduler_skips_entry_on_nse_holiday -v
```
Expected: `ImportError` or `AssertionError`

- [ ] **Step 3: Add `should_skip_trading` to `event_filter.py`**

Import `is_nse_trading_day` from `market_calendar`. `should_skip_trading(d)` returns `True` if `not is_nse_trading_day(d)` OR `check_event_risk(d) is not None`.

- [ ] **Step 4: Wire into `scheduler.py`**

In `check_scheduled_triggers`, import `should_skip_trading` from `event_filter`. At the top of the function body (after the missed-window guard block), add:

```python
if should_skip_trading(today):
    logger.info(f"[Scheduler] {today} is a non-trading day. All triggers skipped.")
    return state
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_event_filter.py tests/test_scheduler.py -v
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add event_filter.py scheduler.py tests/test_event_filter.py tests/test_scheduler.py
git commit -m "fix: skip all order triggers on NSE holidays and high-impact event days"
```

---

## Task 3: Guard LSTM and Hybrid Engine Against Missing Weights

**Files:**
- Modify: `nifty_direction_engine/lstm_predictor.py`
- Modify: `nifty_direction_engine/hybrid_engine.py`
- Create: `tests/test_lstm_predictor.py`
- Create: `tests/test_hybrid_engine.py`

**Interfaces:**
- Produces:
  - `LSTMDirectionPredictor.__init__` raises `RuntimeError("LSTM weights file not found: <path>. Cannot trade without trained weights.")` when the weights file is absent
  - `HybridSignalEngine.generate_signal()` catches `RuntimeError` from either predictor init and returns `{"final_bias": "SIDEWAYS_NO_TRADE", "reason": "Model weights unavailable: <msg>"}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lstm_predictor.py
import pytest
from nifty_direction_engine.lstm_predictor import LSTMDirectionPredictor

def test_lstm_raises_when_weights_missing(tmp_path):
    missing = str(tmp_path / "nonexistent_weights.pt")
    with pytest.raises(RuntimeError, match="LSTM weights file not found"):
        LSTMDirectionPredictor(weights_path=missing)

def test_lstm_loads_successfully_when_weights_present():
    import os
    weights = "nifty_direction_engine/lstm_nifty_weights.pt"
    if not os.path.exists(weights):
        pytest.skip("Weights file not present in this env")
    predictor = LSTMDirectionPredictor(weights_path=weights)
    assert predictor is not None
```

```python
# tests/test_hybrid_engine.py
import pandas as pd
import numpy as np
from unittest.mock import patch
from nifty_direction_engine.hybrid_engine import HybridSignalEngine

def _make_df(n=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    base = 24000.0
    closes = base + np.cumsum(np.random.normal(0, 5, n))
    return pd.DataFrame({
        "timestamps": idx, "open": closes*0.999,
        "high": closes*1.001, "low": closes*0.999, "close": closes,
        "volume": np.ones(n) * 1000
    })

def test_generate_signal_returns_no_trade_when_lstm_weights_missing(tmp_path):
    missing = str(tmp_path / "no_weights.pt")
    with patch("nifty_direction_engine.hybrid_engine.WEIGHTS_PATH", missing):
        engine = HybridSignalEngine(use_kronos=False, use_lstm=True,
                                    weights_path=missing)
        result = engine.generate_signal(_make_df())
    assert result["final_bias"] == "SIDEWAYS_NO_TRADE"
    assert "weights" in result["reason"].lower() or "unavailable" in result["reason"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_lstm_predictor.py tests/test_hybrid_engine.py -v
```
Expected: both fail (predictor currently silently uses random weights)

- [ ] **Step 3: Modify `lstm_predictor.py`**

In `LSTMDirectionPredictor.__init__`, when `weights_path` is provided but `Path(weights_path).exists()` is `False`, raise:
```python
raise RuntimeError(f"LSTM weights file not found: {weights_path}. Cannot trade without trained weights.")
```
Remove the fallback `self.model.eval()` branch that ran with random weights.

- [ ] **Step 4: Modify `hybrid_engine.py`**

Wrap the `LSTMDirectionPredictor(weights_path=path)` constructor call in `__init__` in a `try/except RuntimeError` block. On `RuntimeError`, set `self.lstm = None` and store the error message in `self._lstm_error: str`. In `generate_signal`, if `self.lstm is None` and `self._lstm_error`, immediately return:
```python
{"final_bias": "SIDEWAYS_NO_TRADE", "reason": f"Model weights unavailable: {self._lstm_error}"}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_lstm_predictor.py tests/test_hybrid_engine.py -v
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add nifty_direction_engine/lstm_predictor.py nifty_direction_engine/hybrid_engine.py \
        tests/test_lstm_predictor.py tests/test_hybrid_engine.py
git commit -m "fix: abort signal generation when LSTM weights are missing instead of trading on random weights"
```

---

## Task 4: Remove Synthetic Data from Live Order Path

**Files:**
- Modify: `data_provider.py`
- Modify: `monthly_engine.py`
- Modify: `daily_engine.py`
- Modify: `tests/test_data_provider.py`

**Interfaces:**
- Produces:
  - `DataProvider.fetch_daily(symbol, days, use_cache, allow_synthetic=False)` — raises `RuntimeError("No market data available for <symbol>: Kite and yfinance both failed.")` when `allow_synthetic=False` and both sources fail
  - `DataProvider.fetch_intraday_5m(symbol, days, use_cache, allow_synthetic=False)` — same contract
  - `DataProvider._generate_synthetic_daily` and `_generate_synthetic_5m` are still present but only callable when `allow_synthetic=True`
  - `monthly_engine.run_monthly_analysis` skips an asset (logs warning, sends Telegram) instead of using `FallbackPredictor`
  - `daily_engine.run_intraday_entry` returns `{}` early if `RuntimeError` is raised by data fetch

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_data_provider.py
import pytest
from unittest.mock import patch, MagicMock
from data_provider import DataProvider

def test_fetch_daily_raises_when_both_sources_fail_and_synthetic_disabled():
    dp = DataProvider()
    dp.kite = None  # force Kite path skip
    with patch("data_provider.yf.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = __import__("pandas").DataFrame()
        with pytest.raises(RuntimeError, match="No market data available"):
            dp.fetch_daily("^NSEI", days=30, allow_synthetic=False)

def test_fetch_daily_returns_synthetic_when_explicitly_allowed():
    dp = DataProvider()
    dp.kite = None
    with patch("data_provider.yf.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = __import__("pandas").DataFrame()
        result = dp.fetch_daily("^NSEI", days=30, allow_synthetic=True)
    assert len(result) > 0

def test_fetch_intraday_5m_raises_when_both_fail():
    dp = DataProvider()
    dp.kite = None
    with patch("data_provider.yf.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = __import__("pandas").DataFrame()
        with pytest.raises(RuntimeError, match="No market data available"):
            dp.fetch_intraday_5m("^NSEI", days=3, allow_synthetic=False)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_data_provider.py -v -k "raises or synthetic"
```
Expected: FAIL (currently returns synthetic rows instead of raising)

- [ ] **Step 3: Modify `data_provider.py`**

Add `allow_synthetic: bool = False` parameter to `fetch_daily` and `fetch_intraday_5m`. Replace the final fallback call to `self._generate_synthetic_*` with:
```python
if allow_synthetic:
    return self._generate_synthetic_*()
raise RuntimeError(f"No market data available for {symbol}: Kite and yfinance both failed.")
```

- [ ] **Step 4: Modify `monthly_engine.py`**

Remove `FallbackPredictor` class entirely. In `run_monthly_analysis`, wrap the per-asset block in `try/except RuntimeError as e` — on error, log it, send a Telegram warning for that asset (`f"⚠️ Skipping {name}: {e}"`), and `continue` to the next asset.

Also wrap the Kronos `predictor.predict()` call: on any exception, log and `continue` (skip the asset) rather than falling back to `FallbackPredictor().predict()`.

- [ ] **Step 5: Modify `daily_engine.py`**

In `run_intraday_entry`, wrap the `dp.fetch_intraday_5m(...)` call in `try/except RuntimeError as e`. On error, log it, send a Telegram alert (`f"⚠️ Entry aborted: {e}"`), and `return {}`.

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_data_provider.py tests/test_daily_engine.py -v
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add data_provider.py monthly_engine.py daily_engine.py tests/test_data_provider.py
git commit -m "fix: raise RuntimeError instead of using synthetic data when market data is unavailable"
```

---

## Task 5: Fix Trade Ledger Symbol Bug and EOD Quantity Bug

**Files:**
- Modify: `zerodha_executor.py`
- Modify: `tests/test_zerodha_executor.py`

**Interfaces:**
- Consumes: nothing new
- Produces:
  - `_record_trade_ledger`: each row written uses `leg["tradingsymbol"]` directly, not a variable that may carry a prior leg's value
  - `execute_eod_squareoff`: the BUY quantity placed is `leg.get("quantity")` (set at entry time) — never recomputed from `self.lots`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_zerodha_executor.py
from unittest.mock import patch
from zerodha_executor import ZerodhaOrderExecutor

def test_ledger_records_correct_symbol_per_leg(tmp_path):
    with patch("zerodha_executor.ORDER_BOOK_FILE", tmp_path / "ob.json"), \
         patch("zerodha_executor.ACTIVE_ORDERS_FILE", tmp_path / "ao.json"), \
         patch("zerodha_executor.TRADE_LEDGER_CSV", tmp_path / "ledger.csv"), \
         patch("zerodha_executor.ZerodhaOrderExecutor._is_market_hours", return_value=True):
        executor = ZerodhaOrderExecutor(mode="PAPER", lots=1)
        executor._lot_size = 75
        pos = {
            "date": "2025-06-01",
            "legs": [
                {"action": "SELL", "strike": 24400, "type": "CE",
                 "tradingsymbol": "NIFTY25JUN24400CE",
                 "entry_premium": 50.0, "exit_premium": 10.0,
                 "quantity": 75, "status": "CLOSED", "points_gained": 40.0},
                {"action": "SELL", "strike": 24000, "type": "PE",
                 "tradingsymbol": "NIFTY25JUN24000PE",
                 "entry_premium": 45.0, "exit_premium": 9.0,
                 "quantity": 75, "status": "CLOSED", "points_gained": 36.0},
            ]
        }
        executor._record_trade_ledger(pos)

    import pandas as pd
    df = pd.read_csv(tmp_path / "ledger.csv")
    assert len(df) == 2
    assert "NIFTY25JUN24400CE" in df["tradingsymbol"].values
    assert "NIFTY25JUN24000PE" in df["tradingsymbol"].values

def test_eod_squareoff_uses_leg_quantity_not_config_lots(tmp_path):
    with patch("zerodha_executor.ORDER_BOOK_FILE", tmp_path / "ob.json"), \
         patch("zerodha_executor.ACTIVE_ORDERS_FILE", tmp_path / "ao.json"), \
         patch("zerodha_executor.TRADE_LEDGER_CSV", tmp_path / "ledger.csv"), \
         patch("zerodha_executor.ZerodhaOrderExecutor._is_market_hours", return_value=True), \
         patch("zerodha_executor.send_telegram_message"):
        # Entry was 1 lot (qty=75), but config says lots=2 now
        executor = ZerodhaOrderExecutor(mode="PAPER", lots=2)
        executor._lot_size = 75
        pos = {
            "date": "2025-06-01",
            "legs": [
                {"action": "SELL", "strike": 24400, "type": "CE",
                 "tradingsymbol": "NIFTY25JUN24400CE",
                 "entry_premium": 50.0, "exit_premium": 20.0,
                 "quantity": 75,   # entered with 1 lot
                 "status": "ACTIVE"}
            ]
        }
        result = executor.execute_eod_squareoff(pos)
    assert result["closed_orders"][0]["quantity"] == 75  # must be entry qty, not 2*75=150
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_zerodha_executor.py::test_ledger_records_correct_symbol_per_leg \
                 tests/test_zerodha_executor.py::test_eod_squareoff_uses_leg_quantity_not_config_lots -v
```
Expected: both FAIL

- [ ] **Step 3: Fix `_record_trade_ledger` in `zerodha_executor.py`**

In the `for leg in pos.get("legs", [])` loop inside `_record_trade_ledger`, ensure the dict written to `rows` uses `leg.get("tradingsymbol", "")` directly (not a variable declared outside the loop). The bug is that the variable was accidentally shadowing itself across legs — verify and correct the loop so each row is fully built from its own `leg`.

- [ ] **Step 4: Fix `execute_eod_squareoff` in `zerodha_executor.py`**

Remove the line `total_qty = self.lots * lot_size` at the top of the method. Instead, for each leg, get the quantity as `qty = leg.get("quantity")` — if `None`, log an error and skip that leg. Set `order_record["quantity"] = qty`. Remove the fallback `leg.get("quantity", total_qty)` pattern.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_zerodha_executor.py -v
```
Expected: all pass including the two new tests

- [ ] **Step 6: Commit**

```bash
git add zerodha_executor.py tests/test_zerodha_executor.py
git commit -m "fix: ledger writes correct symbol per leg; EOD squareoff uses entry quantity not config lots"
```

---

## Task 6: Fix Existing Executor Tests Broken by Market-Hours Guard

**Files:**
- Modify: `tests/test_zerodha_executor.py`

**Context:** Task introduced `_is_market_hours()` which blocks all three execute methods when called outside IST market hours (i.e., always in CI). All existing paper-trade tests in `test_zerodha_executor.py::test_executor_paper_trade_entry_and_exit` now return `OUTSIDE_MARKET_HOURS` instead of `COMPLETE`.

**Interfaces:**
- Consumes: `ZerodhaOrderExecutor._is_market_hours` static method
- Produces: existing test patching `_is_market_hours` to return `True`

- [ ] **Step 1: Run existing tests to observe the failure**

```bash
python -m pytest tests/test_zerodha_executor.py::test_executor_paper_trade_entry_and_exit -v
```
Expected: FAIL with `assert "OUTSIDE_MARKET_HOURS" == "COMPLETE"`

- [ ] **Step 2: Patch `_is_market_hours` in the existing test**

In `test_executor_paper_trade_entry_and_exit`, add `patch("zerodha_executor.ZerodhaOrderExecutor._is_market_hours", return_value=True)` as an additional context manager wrapping all three sub-tests (entry, SL exit, EOD squareoff). The patch makes the guard a no-op during testing.

- [ ] **Step 3: Add a dedicated test that the guard blocks outside hours**

```python
def test_execute_entry_blocked_outside_market_hours():
    with patch("zerodha_executor.ZerodhaOrderExecutor._is_market_hours", return_value=False), \
         patch("zerodha_executor.send_telegram_message"):
        executor = ZerodhaOrderExecutor(mode="PAPER", lots=1)
        executor._lot_size = 75
        pos = {"legs": [{"tradingsymbol": "NIFTY25JUN24400CE",
                         "action": "SELL", "type": "CE", "strike": 24400,
                         "entry_premium": 50.0}]}
        result = executor.execute_entry(pos)
    assert result["status"] == "OUTSIDE_MARKET_HOURS"
    assert result["orders"] == []
```

- [ ] **Step 4: Run all executor tests**

```bash
python -m pytest tests/test_zerodha_executor.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tests/test_zerodha_executor.py
git commit -m "fix: patch market-hours guard in executor tests; add dedicated outside-hours test"
```

---

## Task 7: Daily Max-Loss Kill Switch

**Files:**
- Modify: `config.py`
- Modify: `daily_engine.py`
- Modify: `zerodha_executor.py`
- Modify: `tests/test_daily_engine.py`

**Interfaces:**
- Consumes: `TRADE_LEDGER_CSV` path from `zerodha_executor.py`
- Produces:
  - `Config.max_daily_loss_rs: float` — loaded from env var `MAX_DAILY_LOSS_RS` (default `float("inf")`)
  - `get_todays_pnl_rs() -> float` in `daily_engine.py` — reads `trade_ledger.csv`, filters to today's date, sums `pnl_rs`; returns `0.0` if file absent
  - `run_intraday_entry` calls `get_todays_pnl_rs()` before placing orders; if result ≤ `-abs(cfg.max_daily_loss_rs)`, sends a Telegram alert and returns `{"status": "KILL_SWITCH_TRIGGERED"}` without trading

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_daily_engine.py
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

def test_get_todays_pnl_rs_returns_zero_when_no_ledger(tmp_path):
    from daily_engine import get_todays_pnl_rs
    with patch("daily_engine.TRADE_LEDGER_CSV", tmp_path / "ledger.csv"):
        assert get_todays_pnl_rs() == 0.0

def test_get_todays_pnl_rs_sums_todays_rows(tmp_path):
    from daily_engine import get_todays_pnl_rs
    import datetime
    today = str(datetime.date.today())
    df = pd.DataFrame([
        {"date": today, "pnl_rs": -5000.0},
        {"date": today, "pnl_rs": -3000.0},
        {"date": "2020-01-01", "pnl_rs": 99999.0},  # old row, must be ignored
    ])
    ledger = tmp_path / "ledger.csv"
    df.to_csv(ledger, index=False)
    with patch("daily_engine.TRADE_LEDGER_CSV", ledger):
        assert get_todays_pnl_rs() == pytest.approx(-8000.0)

def test_run_intraday_entry_aborts_when_kill_switch_triggered(tmp_path):
    import datetime
    today = str(datetime.date.today())
    df = pd.DataFrame([{"date": today, "pnl_rs": -15000.0}])
    ledger = tmp_path / "ledger.csv"
    df.to_csv(ledger, index=False)

    with patch("daily_engine.TRADE_LEDGER_CSV", ledger), \
         patch("daily_engine.Config.load") as mock_cfg, \
         patch("daily_engine.send_telegram_message"), \
         patch("daily_engine.DataProvider"):
        mock_cfg.return_value.max_daily_loss_rs = 10000.0
        mock_cfg.return_value.auto_trade = False
        from daily_engine import run_intraday_entry
        result = run_intraday_entry(send_telegram=False)
    assert result.get("status") == "KILL_SWITCH_TRIGGERED"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_daily_engine.py -v -k "pnl_rs or kill_switch"
```
Expected: `ImportError` or `AttributeError`

- [ ] **Step 3: Add `max_daily_loss_rs` to `config.py`**

Add field `max_daily_loss_rs: float = float("inf")` to the `Config` dataclass. In `Config.load()`, parse it:
```python
max_daily_loss_rs=float(os.getenv("MAX_DAILY_LOSS_RS", "inf"))
```

- [ ] **Step 4: Add `get_todays_pnl_rs()` to `daily_engine.py`**

Import `TRADE_LEDGER_CSV` from `zerodha_executor` at the top of `daily_engine.py`. Implement:
```python
def get_todays_pnl_rs() -> float:
```
Read CSV with `pd.read_csv`, filter rows where `date == str(datetime.now().date())`, return `float(df["pnl_rs"].sum())`. Return `0.0` on any exception or empty file.

- [ ] **Step 5: Wire kill switch at the top of `run_intraday_entry`**

After loading `cfg = Config.load()`, before any data fetch:
```python
todays_pnl = get_todays_pnl_rs()
if todays_pnl <= -abs(cfg.max_daily_loss_rs):
    msg = f"🛑 Daily loss limit hit (₹{todays_pnl:,.0f}). Kill switch engaged — no new entries today."
    logger.warning(f"[DailyEngine] {msg}")
    send_telegram_message(f"🛑 <b>KILL SWITCH:</b> {msg}")
    return {"status": "KILL_SWITCH_TRIGGERED"}
```

- [ ] **Step 6: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add config.py daily_engine.py tests/test_daily_engine.py
git commit -m "feat: add daily max-loss kill switch via MAX_DAILY_LOSS_RS env var"
```

---

## Final Verification

- [ ] **Run the full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: 0 failures, 0 errors

- [ ] **Verify no synthetic data reaches order path in default config**

```bash
python -c "
from data_provider import DataProvider
dp = DataProvider()
dp.kite = None
import pandas as pd
import unittest.mock as m
with m.patch('data_provider.yf.Ticker') as yf:
    yf.return_value.history.return_value = pd.DataFrame()
    try:
        dp.fetch_intraday_5m('^NSEI', days=3)
        print('FAIL: should have raised')
    except RuntimeError as e:
        print(f'PASS: raised RuntimeError: {e}')
"
```

- [ ] **Verify holiday guard fires correctly**

```bash
python -c "
from datetime import date
from market_calendar import is_nse_trading_day
assert is_nse_trading_day(date(2025, 1, 26)) == False, 'Republic Day should be blocked'
assert is_nse_trading_day(date(2025, 1, 27)) == True, 'Normal Monday should pass'
print('PASS: holiday guard working')
"
```

- [ ] **Final commit**

```bash
git add .
git commit -m "chore: production hardening complete — all 7 tasks done"
```
