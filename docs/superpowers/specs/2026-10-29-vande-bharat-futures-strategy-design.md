# Strategy Specification: Vande Bharat Intraday Futures Strategy

## 1. Executive Summary & Intent
The **Vande Bharat Strategy** (वंदे भारत स्ट्रेटजी) is a disciplined, trend-following intraday breakout strategy designed for Indian equity futures (NSE F&O universe). It was formulated by Kushal Vaishnav to eliminate opening-bell FOMO, exploit high institutional Open Interest (OI) concentration, and capture explosive momentum moves with micro stop-losses and favorable risk-to-reward ratios (typically 1:3 to 1:10+).

This module is designed to reside in a completely decoupled package (`vande_bharat_desk/`), ensuring **zero cross-talk or code coupling** with the existing Kronos foundation model and the other two strategies in this codebase (`daily_engine.py` and `monthly_engine.py`).

---

## 2. Answers to Core Feasibility Questions

### Q1. Is this strategy deterministic?
**Yes, the strategy is mathematically deterministic.**
1. **Core Price Action & Execution (100% Deterministic)**:
   - Timeframe: Fixed 5-minute candles.
   - First 15-minute freeze: Candlestick periods 9:15–9:20, 9:20–9:25, and 9:25–9:30 AM IST are strictly ignored for entries.
   - Thresholds: Previous Day High (PDH) / Previous Day Low (PDL) and First 15-Minute Range ($ORH_{15}$ / $ORL_{15}$ formed between 09:15–09:30 AM IST) are precise mathematical values. Price must trade above $\max(\text{PDH}, ORH_{15})$ for longs, or below $\min(\text{PDL}, ORL_{15})$ for shorts.
   - Pattern: Inside Bar geometry ($High(C_2) \le High(C_1)$ and $Low(C_2) \ge Low(C_1)$) and volume contraction ($Volume(C_2) < Volume(C_1)$) are discrete binary conditions.
   - Exits: 10 EMA cross-under/cross-over and fixed 3:15 PM square-off are unambiguous mathematical triggers.
   - Zero subjective discretion or non-deterministic ML/LLM elements are involved.

2. **Scanner & Selection Determinism**:
   - **Live Mode**: 100% deterministic using real-time NSE/Shoonya pre-market advance-decline ratios and 9:15–9:30 AM $\Delta OI$ calculations.
   - **Backtesting Mode**: Deterministic with one necessary proxy: brokers (including Shoonya and Zerodha) do not archive historical 9:07 AM pre-market snapshot tick dumps. For historical backtesting, we use the 9:15–9:30 AM directional return of NIFTY 50 as the deterministic market breadth proxy (Long if Nifty > 0, Short if Nifty < 0).

---

### Q2. Can it be tested with actual data from Shoonya, in Futures?
**Yes, Shoonya (Finvasia `NorenRestApi`) provides the necessary infrastructure for Futures backtesting and execution**, with specific considerations:

1. **Shoonya Data Endpoints**:
   - Shoonya provides 5-minute historical candlestick data via `get_time_price_series(exchange='NFO', token=..., starttime=..., endtime=..., interval=5)`.
   - The response includes `into` (Open), `inth` (High), `intl` (Low), `intc` (Close), `intv` (Volume), and `intoi` (Open Interest).
2. **Futures Contract Handling & Mapping**:
   - Stock futures contracts are monthly instruments expiring on the last Thursday of each month.
   - Shoonya publishes a daily symbol master (`NFO_symbols.txt`).
   - The backtester dynamically resolves near-month liquid future tokens for each stock (e.g. `RELIANCE24NOVFUT`) and transitions to next-month contracts during expiry week.
3. **Lot Size & Risk Constraints**:
   - Futures require trading in standardized lot sizes (e.g., RELIANCE = 250, TCS = 175).
   - Position sizing must enforce: $\text{Lots} = \lfloor \frac{\text{RPT}}{(\text{Entry} - \text{SL}) \times \text{LotSize}} \rfloor$. If risk for 1 lot exceeds RPT, the trade is skipped.

---

## 3. Strategy Architecture & Execution Rules

```
                      ┌────────────────────────────────────────┐
                      │ 09:07 AM: Pre-Market Breadth Filter    │
                      │  Advances > Declines  ──> Bias = LONG  │
                      │  Declines > Advances  ──> Bias = SHORT │
                      │  ~50 / 50 Balanced    ──> Bias = BOTH  │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ 09:15 - 09:30 AM: Volatility Burnout   │
                      │ Avoid first 3 candles (Station Rush)   │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ 09:30 AM: OI Spurt Scanner (F&O List)  │
                      │ Identify Top 5 stocks with largest     │
                      │ Open Interest surge in first 15 mins   │
                      └───────────────────┬────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
   [ LONG SETUP: Above PDH & ORH15 ]         [ SHORT SETUP: Below PDL & ORL15 ]
  1. Price > max(PDH, 15m High)             1. Price < min(PDL, 15m Low)
  2. Mother Candle C1: Green                2. Mother Candle C1: Red
  3. Inside Candle C2: Red                  3. Inside Candle C2: Green
     - High(C2) <= High(C1)                    - High(C2) <= High(C1)
     - Low(C2) >= Low(C1)                      - Low(C2) >= Low(C1)
  4. Volume: Vol(C2) < Vol(C1)              4. Volume: Vol(C2) < Vol(C1)
  5. Entry: Buy Stop at High(C2)            5. Entry: Sell Stop at Low(C2)
  6. Stop-Loss: Low(C2) - buffer            6. Stop-Loss: High(C2) + buffer
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ Trailing & Exit Rules                  │
                      │ 1. Micro SL hit  ──> Instant Exit      │
                      │ 2. 10 EMA Trail: Close across 10 EMA   │
                      │    & break extreme  ──> Trend Exit     │
                      │ 3. 03:15 PM IST  ──> Mandatory Square Off
                      └────────────────────────────────────────┘
```

### 3.1 Step 1: Universe & Selection
- **Universe**: Active NSE F&O Stock Futures universe (~200–225 stocks).
- **Time Check**: 9:30 AM IST (after 3 full 5-minute candles close).
- **Filter A (Market Breadth)**:
  - If Advances > Declines in F&O: Only allow Long setups.
  - If Declines > Advances in F&O: Only allow Short setups.
  - If approximately 50/50: Allow both Long and Short setups.
- **Filter B (OI Spurt Ranking)**:
  - Rank F&O futures contracts by % change in Open Interest ($\frac{OI_{9:30} - OI_{9:15}}{OI_{9:15}}$).
  - Select Top 5 contracts exhibiting highest positive institutional commitment.
- **Filter C (Dual Level Breakout Filter - PDH/PDL + 15m Opening Range)**:
  - Long candidates must be trading **above BOTH Previous Day High (PDH) AND the First 15-Minute High ($ORH_{15}$)**:
    $$\text{Price} > \max(\text{PDH}, ORH_{15})$$
  - Short candidates must be trading **below BOTH Previous Day Low (PDL) AND the First 15-Minute Low ($ORL_{15}$)**:
    $$\text{Price} < \min(\text{PDL}, ORL_{15})$$
  *(Where $ORH_{15} = \max(\text{High}_{9:15-9:30})$ and $ORL_{15} = \min(\text{Low}_{9:15-9:30})$).*

### 3.2 Step 2: 5-Minute Entry Trigger
- **Long Side (Above $\max(\text{PDH}, ORH_{15})$)**:
  - Trigger Candle $C_1$: Green candle (Close > Open).
  - Inside Candle $C_2$: Red candle (Close < Open).
    - $High(C_2) \le High(C_1)$ and $Low(C_2) \ge Low(C_1)$.
    - Volume condition: $Volume(C_2) < Volume(C_1)$ (indicates low-volume healthy retracement / supply drying up).
  - Entry Trigger: Next bar breaks above $High(C_2)$.
  - Stop Loss: Placed at $Low(C_2) \times (1 - 0.0005)$ (micro-SL).
- **Short Side (Below $\min(\text{PDL}, ORL_{15})$)**:
  - Trigger Candle $C_1$: Red candle (Close < Open).
  - Inside Candle $C_2$: Green candle (Close > Open).
    - $High(C_2) \le High(C_1)$ and $Low(C_2) \ge Low(C_1)$.
    - Volume condition: $Volume(C_2) < Volume(C_1)$ (indicates low-volume healthy retracement / demand drying up).
  - Entry Trigger: Next bar breaks below $Low(C_2)$.
  - Stop Loss: Placed at $High(C_2) \times (1 + 0.0005)$ (micro-SL).

### 3.3 Step 3: Risk Management
- **Risk Per Day (RPD)**: 1.0% of trading capital.
- **Risk Per Trade (RPT)**: $\text{RPD} / \text{Max Daily Trades}$ (e.g. 4 trades = 0.25% capital per trade).
- **Position Sizing**:
  $$\text{Per Share Risk} = |\text{Entry Price} - \text{SL Price}|$$
  $$\text{Lots} = \left\lfloor \frac{\text{RPT}}{\text{Per Share Risk} \times \text{Contract Lot Size}} \right\rfloor$$
  - If calculated $\text{Lots} < 1$, skip the trade (prevents risk violation).

### 3.4 Step 4: Exit & Trailing Engine
1. **Initial Stop-Loss**: Immediate limit-stop in order book at $SL$.
2. **Dynamic 10 EMA Trailing Exit**:
   - Indicator: 10-period Exponential Moving Average on the 5-minute timeframe.
   - Long Trade: If a 5-minute bar closes *below* 10 EMA, mark that bar's Low as the exit trigger. When price violates that Low, close the position.
   - Short Trade: If a 5-minute bar closes *above* 10 EMA, mark that bar's High as the exit trigger. When price violates that High, close the position.
3. **Time-Based Exit (Square-off)**:
   - Mandatory exit at **3:15 PM IST** (intraday F&O cutoff).

---

## 4. Decoupled Codebase Structure

The new strategy is strictly isolated in its own package directory:

```text
Sharads_monthly_option_selling_desk/
│
├── daily_engine.py                    # Strategy 1 (Untouched)
├── monthly_engine.py                  # Strategy 2 (Untouched)
│
├── vande_bharat_desk/                 # Strategy 3 (Completely Isolated)
│   ├── __init__.py
│   ├── config.py                      # Desk parameters (RPD, RPT, EMA period, paths)
│   ├── shoonya_client.py              # Shoonya API authentication, token map, and historical fetcher
│   ├── scanner.py                     # 9:30 AM OI spurt ranker and PDH/PDL filter
│   ├── strategy.py                    # Pure mathematical price action & signal generator
│   ├── backtester.py                  # Event-driven intraday futures backtest simulation engine
│   └── runner.py                      # CLI entrypoint for backtesting & live scanning
│
├── tests/
│   └── test_vande_bharat.py           # Unit tests (Mock Shoonya data, pattern detection, sizing)
└── docs/superpowers/specs/
    └── 2026-10-29-vande-bharat-futures-strategy-design.md
```

---

## 5. Summary of Key Implementation Deliverables

1. **Shoonya Client & Data Cache**:
   - Seamless token master loading (`NFO_symbols.txt`).
   - Caching 5-minute futures candles to local Parquet files in `data_cache/shoonya_futures/`.
2. **Deterministic Backtester**:
   - Evaluates tick-by-tick / candle-by-candle simulated execution.
   - Accurately accounts for contract lot sizes, slippage buffer, STT, and exchange transaction fees.
3. **Comprehensive Unit Testing**:
   - Verifies exact inside-bar detection, volume contraction logic, 10 EMA trailing transitions, and position size calculations.
