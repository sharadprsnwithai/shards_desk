# Monthly Multi-Asset LSTM & Desk Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a dedicated multi-asset positional LSTM engine trained on daily candles across 6 assets (NIFTY 50, BANK NIFTY, SENSEX, RELIANCE, TCS, HDFCBANK), integrate it into `monthly_engine.py` as an ensemble with Kronos, and update NIFTY lot size to 65 across reporting without touching `daily_engine.py`.

**Architecture:** A standalone `monthly_direction_engine/` module extracts scale-invariant daily features (returns, ATR, RSI, moving average deviations, realized volatility) and runs an LSTM predicting forward 30-day price returns. `monthly_engine.py` loads this model alongside the Kronos foundation model to form an ensemble directional bias and boundary containment check for all 6 core assets.

**Tech Stack:** Python 3.10+, PyTorch, Pandas, NumPy, SciPy, Zerodha Kite Connect / yfinance.

**Spec:** `docs/superpowers/specs/2026-10-28-monthly-multi-asset-lstm-design.md`

## Global Constraints
- `daily_engine.py` and `nifty_direction_engine/` must NOT be touched or modified.
- NIFTY lot size must be updated from 25 to 65 in reporting and lot configs.
- Memory consumption of the trained LSTM must be < 5 MB and fit easily inside 1GB VPS limits.
- Must operate with fallback resiliency when Kite API tokens are offline.

## Review Focus
- Asset scale differences (Index at 24,000 vs stock at 1,700) must produce equivalent normalized feature distributions.
- Handling insufficient historical candle length (< 60 daily bars) gracefully by returning a neutral return prediction.
- Bank Nifty token mapping must match Zerodha's NSE token (`260105`).
- Telegram reports and console displays must properly display lot sizes and multi-asset legs.
- All existing and new automated unit tests must pass with 100% green status.

---

### Task 1: Update NIFTY Lot Size to 65 & Register Bank Nifty Instrument Token

**Files:**
- Modify: `data_provider.py:51-57`
- Modify: `telegram_notifier.py:204-208`
- Modify: `backtest_intraday_roi.py:26`
- Modify: `backtest_sl10_target50.py:25`
- Modify: `backtest_sl50_target80.py:25`
- Modify: `backtest_variations.py:23`
- Test: `tests/test_data_provider.py`

**Interfaces:**
- Consumes: `KITE_INSTRUMENT_MAP` in `data_provider.py`
- Produces: `^NSEBANK` entry in `KITE_INSTRUMENT_MAP` with token `260105`, and `lot_size = 65` for Nifty PnL calculations.

- [ ] **Step 1: Write the failing test for `^NSEBANK` instrument mapping**

Add to `tests/test_data_provider.py`:
```python
def test_banknifty_instrument_mapping():
    dp = DataProvider()
    assert dp.get_instrument_token("^NSEBANK") == 260105
    assert dp.get_tradingsymbol("^NSEBANK") == "NIFTY BANK"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_provider.py -k test_banknifty_instrument_mapping -v`
Expected: FAIL (token is None).

- [ ] **Step 3: Add `^NSEBANK` to `data_provider.py` and update NIFTY lot sizes to 65**

In `data_provider.py`:
```python
KITE_INSTRUMENT_MAP = {
    "^NSEI": {"tradingsymbol": "NIFTY 50", "exchange": "NSE", "token": 256265},
    "^NSEBANK": {"tradingsymbol": "NIFTY BANK", "exchange": "NSE", "token": 260105},
    "^BSESN": {"tradingsymbol": "SENSEX", "exchange": "BSE", "token": 265},
    "RELIANCE.NS": {"tradingsymbol": "RELIANCE", "exchange": "NSE", "token": 738561},
    "TCS.NS": {"tradingsymbol": "TCS", "exchange": "NSE", "token": 2953217},
    "HDFCBANK.NS": {"tradingsymbol": "HDFCBANK", "exchange": "NSE", "token": 341249}
}
```
In `telegram_notifier.py:205`: Change `lot_size = 25` to `lot_size = 65`.
In backtest files (`backtest_intraday_roi.py`, `backtest_sl10_target50.py`, `backtest_sl50_target80.py`, `backtest_variations.py`): Change NIFTY lot size constant from 25 to 65.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_provider.py -k test_banknifty_instrument_mapping -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data_provider.py telegram_notifier.py backtest_*.py tests/test_data_provider.py
git commit -m "fix: update nifty lot size to 65 and add banknifty instrument token"
```

---

### Task 2: Create Scale-Invariant Feature Extractor for Positional Daily Data

**Files:**
- Create: `monthly_direction_engine/feature_extractor.py`
- Test: `tests/test_monthly_lstm.py`

**Interfaces:**
- Consumes: Daily OHLCV `pd.DataFrame` with columns `["timestamps", "open", "high", "low", "close", "volume"]`
- Produces: `extract_positional_features(df: pd.DataFrame) -> pd.DataFrame` with 7 scale-invariant indicator columns:
  `["norm_ret", "hl_range", "norm_atr", "norm_rsi", "sma20_dev", "sma50_dev", "realized_vol"]`

- [ ] **Step 1: Write the failing test for positional feature extraction**

Create `tests/test_monthly_lstm.py`:
```python
import numpy as np
import pandas as pd
import pytest
from monthly_direction_engine.feature_extractor import extract_positional_features

def test_extract_positional_features():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = np.linspace(20000, 22000, 100) + np.random.normal(0, 50, 100)
    df = pd.DataFrame({
        "timestamps": dates,
        "open": prices * 0.998,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": [100000] * 100
    })
    feats = extract_positional_features(df)
    assert len(feats) == 100
    expected_cols = ["norm_ret", "hl_range", "norm_atr", "norm_rsi", "sma20_dev", "sma50_dev", "realized_vol"]
    for col in expected_cols:
        assert col in feats.columns
    # Check no NaNs after backfilling initial window
    assert not feats[expected_cols].isna().any().any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monthly_lstm.py::test_extract_positional_features -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `extract_positional_features` in `monthly_direction_engine/feature_extractor.py`**

Implement technical indicator functions (RSI, ATR, SMA deviations, realized volatility) that operate strictly on relative price ratios and percentages, and return a clean DataFrame.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monthly_lstm.py::test_extract_positional_features -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add monthly_direction_engine/feature_extractor.py tests/test_monthly_lstm.py
git commit -m "feat: add scale-invariant positional feature extractor for daily data"
```

---

### Task 3: Implement Positional LSTM Model & Inference Predictor

**Files:**
- Create: `monthly_direction_engine/monthly_lstm_model.py`
- Test: `tests/test_monthly_lstm.py`

**Interfaces:**
- Consumes: Features from `extract_positional_features`
- Produces:
  - `MonthlyLSTMModel(input_dim=7, hidden_dim=64, num_layers=2, dropout=0.2)`
  - `MonthlyLSTMPredictor(weights_path=None, device=None)`
  - `MonthlyLSTMPredictor.predict_return(df: pd.DataFrame, seq_len: int = 60) -> float`

- [ ] **Step 1: Write failing test for `MonthlyLSTMPredictor`**

Add to `tests/test_monthly_lstm.py`:
```python
from monthly_direction_engine.monthly_lstm_model import MonthlyLSTMModel, MonthlyLSTMPredictor

def test_monthly_lstm_model_forward():
    model = MonthlyLSTMModel(input_dim=7, hidden_dim=64, num_layers=2)
    dummy_input = torch.randn(4, 60, 7)
    out = model(dummy_input)
    assert out.shape == (4, 1)

def test_monthly_lstm_predictor_inference():
    predictor = MonthlyLSTMPredictor(weights_path=None) # Tests uninitialized / untrained fallback
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    prices = np.linspace(24000, 24500, 80)
    df = pd.DataFrame({
        "timestamps": dates,
        "open": prices, "high": prices*1.01, "low": prices*0.99, "close": prices, "volume": [1000]*80
    })
    pred_ret = predictor.predict_return(df, seq_len=60)
    assert isinstance(pred_ret, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monthly_lstm.py -k test_monthly_lstm -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `MonthlyLSTMModel` and `MonthlyLSTMPredictor` in `monthly_direction_engine/monthly_lstm_model.py`**

Include model checkpoint saving and loading (loading `mean`, `std`, and `model_state_dict`), automatic device mapping (CPU/CUDA), and edge-case handling when input rows < `seq_len`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monthly_lstm.py -k test_monthly_lstm -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add monthly_direction_engine/monthly_lstm_model.py tests/test_monthly_lstm.py
git commit -m "feat: implement MonthlyLSTMModel and MonthlyLSTMPredictor"
```

---

### Task 4: Multi-Asset Training Script & Weights Generation

**Files:**
- Create: `monthly_direction_engine/train_monthly_lstm.py`
- Create: `monthly_direction_engine/monthly_lstm_weights.pt`
- Test: `tests/test_monthly_lstm.py`

**Interfaces:**
- Consumes: `DataProvider.fetch_daily` for all 6 symbols
- Produces:
  - `train_multi_asset_monthly_lstm(epochs=30, batch_size=32, lr=0.001) -> Path`
  - Saved model weights at `monthly_direction_engine/monthly_lstm_weights.pt`

- [ ] **Step 1: Write training dataset creation and execution step**

In `monthly_direction_engine/train_monthly_lstm.py`:
- Fetch daily historical data (up to 1,500 - 2,500 daily bars per asset) for `^NSEI`, `^NSEBANK`, `^BSESN`, `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`.
- Generate sliding sequences `(seq_len=60)` with target `return_30d = (close[t+30] - close[t]) / close[t]`.
- Pool samples from all 6 assets into unified `TensorDataset` and `DataLoader`.
- Train model using MSE Loss, Adam optimizer, and cosine annealing or step LR.
- Save trained model, weights, mean, and std to `monthly_direction_engine/monthly_lstm_weights.pt`.

- [ ] **Step 2: Run training script to generate baseline production weights**

Run: `python monthly_direction_engine/train_monthly_lstm.py`
Expected: Training completes across pooled samples and saves `monthly_lstm_weights.pt` (< 2MB).

- [ ] **Step 3: Add test asserting trained weights load properly**

Add to `tests/test_monthly_lstm.py`:
```python
def test_load_trained_weights():
    weights_path = Path("monthly_direction_engine/monthly_lstm_weights.pt")
    assert weights_path.exists()
    predictor = MonthlyLSTMPredictor(weights_path=str(weights_path))
    assert predictor.mean is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monthly_lstm.py::test_load_trained_weights -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add monthly_direction_engine/train_monthly_lstm.py monthly_direction_engine/monthly_lstm_weights.pt tests/test_monthly_lstm.py
git commit -m "feat: train and save multi-asset monthly positional LSTM weights"
```

---

### Task 5: Expand `monthly_engine.py` to 6 Assets with Kronos + LSTM Ensemble

**Files:**
- Modify: `monthly_engine.py:30-140`
- Test: `tests/test_monthly_engine.py`

**Interfaces:**
- Consumes: `KronosPredictor` and `MonthlyLSTMPredictor`
- Produces: `run_monthly_analysis() -> List[Dict[str, Any]]` evaluating all 6 assets with combined ensemble bias, lot sizes, and defined-risk spread strikes.

- [ ] **Step 1: Update `tests/test_monthly_engine.py` with 6-asset mock test**

Ensure test covers multi-asset runs, Kronos + LSTM ensemble outputs, and lot size inclusion.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monthly_engine.py -v`
Expected: FAIL due to missing assets and ensemble fields.

- [ ] **Step 3: Implement multi-asset list and ensemble logic in `monthly_engine.py`**

In `monthly_engine.py`:
- Update `MONTHLY_ASSETS` to include all 6 instruments:
  - `^NSEI` (NIFTY 50, step=50, wing=300, lot_size=65)
  - `^NSEBANK` (BANK NIFTY, step=100, wing=500, lot_size=15)
  - `^BSESN` (SENSEX, step=100, wing=600, lot_size=10)
  - `RELIANCE.NS` (RELIANCE, step=20, wing=100, lot_size=250)
  - `TCS.NS` (TCS, step=20, wing=100, lot_size=175)
  - `HDFCBANK.NS` (HDFCBANK, step=10, wing=50, lot_size=550)
- Load `MonthlyLSTMPredictor(weights_path="monthly_direction_engine/monthly_lstm_weights.pt")`.
- Inside the asset loop:
  - Calculate `lstm_ret = lstm_predictor.predict_return(df)`
  - Combine Kronos trajectory target with `lstm_ret` to determine ensemble directional conviction.
  - Call `determine_option_strategy` and pass `lot_size` in the result dict.
  - Print formatted console summary showing both Kronos and LSTM projections.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_monthly_engine.py -v`
Expected: PASS.

- [ ] **Step 5: Run full test suite across the entire repository**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add monthly_engine.py tests/test_monthly_engine.py
git commit -m "feat: ensemble monthly engine with multi-asset LSTM across all 6 core assets"
```
