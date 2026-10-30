# Design Document: Monthly Multi-Asset LSTM & Desk Update

## 1. Intent & Why
The monthly positional option selling desk (`monthly_engine.py`) currently relies solely on the zero-shot Kronos foundation model and classical diffusion math for NIFTY 50. To increase quantitative robustness, this project introduces a dedicated multi-asset Positional LSTM trained across all 6 core assets (**NIFTY 50, SENSEX, BANK NIFTY, RELIANCE, TCS, HDFCBANK**). It ensembles with Kronos to evaluate 30-day directional trends and breakout risks before recommending defined-risk option structures (Iron Condors, Bull Put Spreads, Bear Call Spreads).

Additionally, this project updates the NIFTY lot size from **25 to 65** across reporting and backtesting modules, while strictly ensuring the intraday desk (`daily_engine.py`) remains untouched.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- **Multi-Asset Monthly LSTM Module (`monthly_direction_engine/`)**:
  - `monthly_lstm_model.py`: Scale-invariant PyTorch LSTM architecture and `MonthlyLSTMPredictor` inference class.
  - `feature_extractor.py`: Prepares scale-invariant technical indicators from daily candles (Log Returns, RSI-14, Normalized ATR, 20/50 SMA deviation, Rolling 20-Day Realized Volatility).
  - `train_monthly_lstm.py`: Downloads 7–10 years of daily historical data for all 6 assets via `DataProvider` (using Zerodha Kite Connect with cached parquet storage), constructs pooled sliding windows (~12,000+ samples), trains the model, and saves weights.
- **Instrument Expansion in `data_provider.py`**:
  - Add `^NSEBANK` (`NIFTY BANK`, NSE token `260105`) to `KITE_INSTRUMENT_MAP`.
- **Monthly Desk Expansion in `monthly_engine.py`**:
  - Expand `MONTHLY_ASSETS` from just NIFTY 50 to all 6 assets with their exchange strike steps, wing widths, and lot sizes.
  - Integrate `MonthlyLSTMPredictor` alongside `KronosPredictor` to form an ensemble signal (Directional agreement, Confidence, and Strategy selection).
- **Lot Size Correction**:
  - Update NIFTY lot size to **65** in `telegram_notifier.py`, console summaries, and relevant backtests.
- **Test Suite**:
  - Add comprehensive unit tests in `tests/test_monthly_lstm.py` covering feature extraction, inference shapes, fallback behavior, and ensemble logic in `monthly_engine.py`.

### Non-Goals
- Modifying `daily_engine.py` or files inside `nifty_direction_engine/` (zero touch).
- Modifying live order placement execution (this desk produces quant analysis, strike levels, and Telegram notifications).

### Anti-Goals
- Adding heavy external frameworks or dependencies outside the existing stack (`torch`, `pandas`, `numpy`, `scipy`).
- Overfitting models to absolute index prices; all features and targets must remain scale-invariant (percentage returns and ratios).

---

## 3. Constraints
- **Resource Constraints**: Must operate reliably within a 1GB RAM + 2GB Swap VPS environment. The trained PyTorch model must remain under 5MB with sub-second inference.
- **Offline / Test Resiliency**: The system must operate cleanly even when live Kite sessions are unavailable, utilizing local parquet cache, fallback predictors, and unit test mocks.

---

## 4. Approaches Considered
- **Approach 1: Unified Multi-Asset Pooled Model (Selected)**
  - A single LSTM trained across all 6 assets with scale-invariant normalized features.
  - *Trade-off*: Requires rigorous scale-invariant normalization so a ₹24,000 index and a ₹1,700 stock train harmoniously.
  - *Why Selected*: Yields >12,000 training windows, eliminating small-sample overfitting, while keeping memory overhead and deployment lean.
- **Approach 2: Asset-Specific Individual Models (6 Separate LSTMs)**
  - *Why Dropped*: High risk of overfitting due to only ~2,000 daily bars per asset, plus 6 separate weight files to version and maintain.
- **Approach 3: Two-Tier Clustered Models (Index Model + Equities Model)**
  - *Why Dropped*: Adds intermediate complexity without statistical advantage over properly normalized multi-asset pooling.

---

## 5. Architectural Specification & How It Works

### A. Data & Features (`monthly_direction_engine/feature_extractor.py`)
For each asset, input sequences use a sliding window of **60 daily candles** (~3 months of trading context):
1. **Normalized Daily Return**: $\ln(C_t / C_{t-1})$
2. **Normalized High-Low Range**: $(High_t - Low_t) / Close_t$
3. **Normalized 14-Day ATR**: $ATR_{14} / Close_t$
4. **14-Day RSI**: Scaled to $[-1.0, 1.0]$ via $(RSI - 50) / 50$
5. **20-Day SMA Deviation**: $(Close_t - SMA_{20}) / SMA_{20}$
6. **50-Day SMA Deviation**: $(Close_t - SMA_{50}) / SMA_{50}$
7. **20-Day Realized Volatility**: Rolling annualized $\sigma$ of log returns

**Target Variable**:
- Forward 30-day return %: $\frac{Close_{t+30} - Close_t}{Close_t}$.

### B. Network Architecture (`MonthlyLSTMModel`)
- Input Dim: 7 features
- Hidden Dim: 64 units, 2 layers, Dropout: 0.2
- Dense projection: 64 -> 32 (ReLU + Dropout 0.2) -> 1 (Linear return output)
- Loss: MSE Loss with Adam optimizer (learning rate ~0.001)

### C. Ensemble Decision Matrix in `monthly_engine.py`
For each of the 6 assets:
- **Kronos**: Forecasts 30-day OHLC path $\rightarrow$ yields $r_{kronos} = (Close_{30} - Close_0) / Close_0$, Support ($Low_{min}$), Resistance ($High_{max}$), and Containment Probability $P_{diff}$.
- **Monthly LSTM**: Evaluates the 60-day historical sequence $\rightarrow$ yields $r_{lstm}$.
- **Ensemble Bias**:
  - `Bullish`: $r_{kronos} \ge +1.0\%$ and $r_{lstm} \ge +0.5\%$
  - `Bearish`: $r_{kronos} \le -1.0\%$ and $r_{lstm} \le -0.5\%$
  - `Neutral / Range-Bound`: Both within $[-1.0\%, +1.0\%]$ or $P_{diff} \ge 60\%$
  - `Conflicting / Divergent`: Opposite signs exceeding deadband $\rightarrow$ flagged with reduced confidence or wider protective strikes.

### D. Asset Configuration in `monthly_engine.py`
```python
MONTHLY_ASSETS = [
    {"symbol": "^NSEI", "name": "NIFTY 50", "step": 50, "wing": 300, "lot_size": 65},
    {"symbol": "^NSEBANK", "name": "BANK NIFTY", "step": 100, "wing": 500, "lot_size": 15},
    {"symbol": "^BSESN", "name": "SENSEX", "step": 100, "wing": 600, "lot_size": 10},
    {"symbol": "RELIANCE.NS", "name": "RELIANCE", "step": 20, "wing": 100, "lot_size": 250},
    {"symbol": "TCS.NS", "name": "TCS", "step": 20, "wing": 100, "lot_size": 175},
    {"symbol": "HDFCBANK.NS", "name": "HDFCBANK", "step": 10, "wing": 50, "lot_size": 550},
]
```

---

## 6. What's Left to the Builder
- Implementing the PyTorch training loop with early stopping / validation loss tracking.
- Generating and committing the baseline trained weights `monthly_lstm_weights.pt` using the Kite/Yahoo cache dataset.
- Updating all unit tests to assert 65 lot size for NIFTY.
