# Strategy Desk Review Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 6 must-fix findings from the 2026-10-31 desk review (daily 4-OTM breathing naked vs hedged, event-day trading, untracked failed square-offs, backtest/live logic mismatch, costless backtest, unguarded zero-premium quotes).

**Architecture:** Two independent subsystems. Phase A hardens `daily_engine.py` + `zerodha_executor.py` (order lifecycle, legs, exit integrity). Phase B makes `sector_momentum_desk/backtester.py` identical in logic to the live scorer/trigger and charges it costs. No new modules, no new dependencies.

**Tech Stack:** Python 3.13, pytest, existing pandas/yfinance/pytz utilities already in repo.

**Spec:** This plan implements the findings in the 2026-10-31 senior-analyst review conversation (must-fix items 1–6, should-fix 7, 8, 11). Findings are copied verbatim into task context below.

## Global Constraints

- Default trading mode stays PAPER (`config.py:intraday_option_selling = "PAPER"`). Nothing in this plan may flip a default to LIVE.
- All Telegram-alert-on-failure behavior must be preserved (errors alert, never silently swallowed).
- Market-hours guard `_is_market_hours()` (9:15–15:30 IST, Mon–Fri) unchanged.
- No new pip dependencies.
- Every task ends with `pytest` green on the touched test file plus the full suite once per phase.
- IST = Asia/Kolkata via `pytz` as existing code does.

## Review Focus

The five failure modes the review surfaced that task tests must pin:

1. **Event day (budget/RBI/Fed)** — `check_event_risk()` returns a note and the desk must place zero legs that morning. Pinned in Task 3.
2. **Rejected EOD square-off order in LIVE** — a leg whose EOD BUY is REJECTED must keep the position file on disk and alert, never `clear_active_position()`. Pinned in Task 4.
3. **Hedge wing executed in the wrong direction** — legs with `"action": "BUY"` must be placed as BUY orders and must be skipped by SL/TP monitor and EOD square-off. Pinned in Tasks 1–2.
4. **Zero / failed option quote** — a leg with `ltp <= 0` must abort that leg (and, if all legs fail, the whole entry), never reach the executor with `sl_premium = 0`. Pinned in Task 8.
5. **Cross-session EMA leak in backtest** — the 10-EMA trail must be computed per trading day only, matching live `check_trailing_exit`. Pinned in Task 5.

---

# Phase A — Daily Desk (`daily_engine.py`, `zerodha_executor.py`)

### Task 1: Hedged legs on daily entry + honest docstring + remove dead import

**Files:**
- Modify: `daily_engine.py:25` (delete dead import), `daily_engine.py:64-182` (docstring + legs), `daily_engine.py:241-269` (monitor skips BUY legs)
- Test: `tests/test_daily_engine.py`

**Context from review:** `determine_intraday_option_strategy` is imported at line 25 and never called; the docstring promises "Bear Call / Bull Put credit spreads / Iron Condor" but the SIDEWAYS branch sells a **naked strangle**. Minimal fix: build the in-repo hedge-wing pattern inline (sold strike ±150), exactly as `options_engine.py:181-206` does, and stop claiming what the code is not. Do **not** route through `determine_intraday_option_strategy` — it computes strikes from day high/low, not the fixed 200-pt OTM rule this desk trades.

**Interfaces:**
- Produces: every SELL leg in `position_payload["legs"]` gains a companion BUY leg object, e.g. sold CE 24500 → companion `{action: "BUY", strike: 24650, type: "CE", "desc": "Hedge Wing"}`. Companion leg has `entry_premium`, no `sl_premium`/`target_premium` (exited only by EOD).

- [ ] **Step 1: Write the failing tests** in `tests/test_daily_engine.py`:

```python
def test_entry_legs_contain_hedge_wings():
    # SIDEWAYS bias must produce 4 legs: SELL CE, BUY CE(+150), SELL PE, BUY PE(-150)
    legs = build_legs_for_test(bias="SIDEWAYS", atm=24000)  # helper calling the refactored leg builder
    actions = [l["action"] for l in legs]
    assert actions.count("SELL") == 2 and actions.count("BUY") == 2
    assert {l["strike"] for l in legs if l["action"] == "BUY"} == {24150 - 150, 24150 + 150}  # matches sold strikes ±150
```

```python
def test_monitor_skips_buy_legs():
    # A BUY leg with an absurd sl_premium must never be flagged by the SL/TP monitor
    # (monitor already skips legs lacking sl_premium — pin that a BUY leg is ignored)
```

  The leg-building block currently inlined in `run_intraday_entry` (lines 114–182) moves to a pure helper `build_legs(bias, atm_strike, target_expiry, quote_fn) -> list[dict]` so the test can inject `quote_fn` returning fixed LTPs — no Kite needed.
- [ ] **Step 2: Run** `pytest tests/test_daily_engine.py -v` — expect FAIL on both new tests.
- [ ] **Step 3: Implement**: helper `build_legs`, wings appended per branch (CE wing = sold +150, PE wing = sold −150), docstring at L64–71 rewritten to describe 2-leg credit spreads with wings, delete `from options_engine import determine_intraday_option_strategy` (L25).
- [ ] **Step 4: Run** `pytest tests/test_daily_engine.py -v` — expect PASS.
- [ ] **Step 5: Commit** `git commit -m "daily: hedged legs via in-repo wing pattern, drop dead import"`

### Task 2: Executor places BUY legs correctly and skips them on square-off

**Files:**
- Modify: `zerodha_executor.py:187-225` (`execute_entry` — use `leg["action"]`), `zerodha_executor.py:280-360` (`execute_sl_tp_exit` — early-return on BUY legs), `zerodha_executor.py:362-434` (`execute_eod_squareoff` — skip BUY legs)
- Test: `tests/test_zerodha_executor.py`

**Context from review:** `execute_entry` currently hardcodes `TRANSACTION_TYPE_SELL` for every leg (line 220). With Task 1's wings that would SELL the hedge too — doubling the naked exposure instead of hedging it. EOD square-off must not BUY-to-close an already-long wing.

**Interfaces:**
- Consumes: leg dicts from Task 1 (`action` ∈ {"SELL","BUY"}).
- Produces: unchanged return shapes (`{"mode", "timestamp", "lots", "quantity", "orders"}`).

- [ ] **Step 1: Write the failing test**:

```python
def test_entry_places_buy_legs_as_buy():
    # PAPER mode, position with 2 SELL + 2 BUY legs ->
    # order book must contain 2 records action="SELL" and 2 records action="BUY"
def test_eod_squareoff_skips_buy_legs():
    # EOD: BUY legs receive no square-off record; status untouched
```

- [ ] **Step 2: Run** `pytest tests/test_zerodha_executor.py -v` — expect FAIL.
- [ ] **Step 3: Implement**: `transaction_type = BUY/SELL from leg action`; `execute_sl_tp_exit` returns `{"status":"SKIPPED_LONG_LEG"}` when `leg.get("action")=="BUY"`; same guard in `execute_eod_squareoff` loop next to the existing `status in ("STOP_LOSS_HIT",...)` skip.
- [ ] **Step 4: Run** `pytest tests/test_zerodha_executor.py -v` — PASS.
- [ ] **Step 5: Commit** `git commit -m "executor: honor leg action; square-off touches SELL legs only"`

### Task 3: Event day = no trade

**Files:**
- Modify: `daily_engine.py:81-83`
- Test: `tests/test_daily_engine.py`

**Context from review:** the event gate prints "⚠️ HIGH IMPACT EVENT TODAY" and then sells naked premium anyway. Highest tail-risk day of the month must be a no-trade day.

**Interfaces:**
- Produces: when `check_event_risk()` returns a non-empty string, `run_intraday_entry` sends one Telegram message and returns `{}` **before** fetching data, building legs, writing the position file, or executing orders.

- [ ] **Step 1: Write the failing test**:

```python
def test_event_day_places_no_trade(monkeypatch):
    monkeypatch.setattr("daily_engine.check_event_risk", lambda: "RBI Policy")
    assert run_intraday_entry(send_telegram=False) == {}
    assert not ACTIVE_POSITION_FILE.exists()
```

- [ ] **Step 2: Run** — expect FAIL (currently proceeds).
- [ ] **Step 3: Implement** — after printing the note, `return {}`. One line of comment: event days bypass the hybrid signal entirely by policy.

- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `git commit -m "daily: hard no-trade on high-impact event days"`

### Task 4: Exit integrity — real fills in ledger, safe position-file clear

**Files:**
- Modify: `daily_engine.py:278-284` (SL branch), `daily_engine.py:415-424` (EOD block)
- Test: `tests/test_daily_engine.py`

**Context from review:** (a) on SL hit the ledger records `points_gained = −0.5×entry` even when the recorded `exit_premium` shows a worse market fill — losses systematically understated; (b) the position file is deleted even when an EOD square-off order was REJECTED, leaving an untracked live short overnight.

**Interfaces:**
- Produces:
  - SL branch: `points_gained = entry_prem - exit_premium` (exit premium = actual fill or current LTP, whatever is recorded — negatives allowed and expected).
  - New tiny helper `verify_all_legs_closed(pos) -> bool`: True iff every SELL leg has `order_record`-backed status COMPLETE (leg status CLOSED/TARGET_HIT/STOP_LOSS_HIT and its order id present). `clear_active_position()` runs only when True; otherwise send a Telegram alert "EOD square-off incomplete — position file retained" and leave the file on disk.

- [ ] **Step 1: Write the failing tests**:

```python
def test_sl_points_derive_from_actual_fill():
    # leg entry 100, sl_premium 150, exit fill 165 -> points_gained == -65, not -50
def test_eod_keeps_file_on_rejected_order(monkeypatch):
    # one leg order status "REJECTED" -> file still exists, alert sent
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** both changes; the REJECTED leg keeps `status = "OPEN"` and is re-attempted on the next `run_intraday_exit` invocation.
- [ ] **Step 4: Run** — PASS, then full suite `pytest`.
- [ ] **Step 5: Commit** `git commit -m "daily: fill-based SL PnL; verified-clear before deleting position state"`

### Task 8: Quote sanity gate before building legs

**Files:**
- Modify: `daily_engine.py` — inside new helper `build_legs` (Task 1)
- Test: `tests/test_daily_engine.py`

**Context from review:** a failed Kite quote returns `(None, 0.0)` and the leg still ships to the executor with `sl_premium = 0` — a stop-less naked leg.

**Interfaces:**
- Consumes: quote_fn from Task 1.
- Produces: any leg whose `ltp <= 0.0` is excluded; if **all** legs for the bias are excluded, `run_intraday_entry` returns `{}` with a Telegram "quotes unavailable" message and no position file write.

- [ ] **Step 1: Write the failing test**: `quote_fn` returning 0.0 → `build_legs(...) == []` and `run_intraday_entry` returns `{}`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** the guard in `build_legs` (one `if ltp <= 0: continue` + count check).
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `git commit -m "daily: abort legs / entry on zero-premium quote"`

---

# Phase B — Momentum Desk (`sector_momentum_desk/`)

### Task 5: Backtest/live parity — per-day EMA and matching volume weight

**Files:**
- Modify: `sector_momentum_desk/backtester.py:90-101` (drop whole-series EMA), `backtester.py:229` (volume source), `backtester.py:416-427` (per-day EMA use)
- Test: `tests/test_sector_momentum.py`

**Context from review:** live weights the score by the **latest single 5m candle volume** (`stock_scorer.py:114-121`) while the backtest uses the **15-minute volume sum** (`backtester.py:228`); live computes the 10-EMA on **same-day candles only** (`entry_trigger.py:297`) while the backtest EMA runs over the continuous multi-month series (`backtester.py:90`), carrying yesterday's regime across the overnight gap. The backtest therefore validates a different strategy than the one that trades.

**Interfaces:**
- Produces: `_backtest_single_day` weights each candidate by `stk_930["Volume"].iloc[-1]` (the 9:25 candle — same as live at 09:30); `self.df_ema10` deleted; EMA inside `_simulate_stock_day` computed as `session_candles["Close"].ewm(span=EMA_TRAIL_PERIOD, adjust=False).mean()`, exit checked at index ≥ EMA_TRAIL_PERIOD (live's `len < EMA_TRAIL_PERIOD` guard).

- [ ] **Step 1: Write the failing test**:

```python
def test_backtest_ema_is_per_day():
    # strip of two days; same intraday prices -> EMA at day2 10:00 must equal
    # the value pandoc gives for day2-only series (no day1 carry-in)
def test_backtest_volume_weight_matches_live():
    # candidate volume == last 5m candle before 09:30, not the 15m sum
```

  (Test assertions: backtester candidate volume equals `stk_930["Volume"].iloc[-1]`, i.e. the last 5m candle before 09:30; and for a synthetic 2-day frame, the trailed exit EMA equals the ewm of the day-2-only series.)
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** the two changes above.
- [ ] **Step 4: Run** `pytest tests/test_sector_momentum.py -v` — PASS.
- [ ] **Step 5: Commit** `git commit -m "backtest: per-day EMA + live-matching volume weight (parity fix)"`

### Task 6: Costs and breakout slippage in the backtest

**Files:**
- Modify: `sector_momentum_desk/config.py` (add constants), `sector_momentum_desk/backtester.py` (PnL lines 393-476 — apply once in each exit branch via one helper)
- Test: `tests/test_sector_momentum.py`

**Context from review:** the 6-month ledger shows many trades grossing +0.0% to +0.5% with zero brokerage/STT/slippage — the edge dies on a cold server before costs. PDH-breakout entries at the pullback extreme are fills a live order cannot get; gate entries almost always pay above touch.

**Interfaces:**
- Produces: two constants in `config.py`:
  - `COST_BPS_ROUNDTRIP: float = 25.0` (≈ brokerage+STT+txn+GST both sides).
  - `BREAKOUT_SLIPPAGE_PCT: float = 0.05` (percent of entry, charged against the trader on gated entries only).
  - One helper `_apply_costs(pnl_pts, entry_price, slippage: bool) -> float` used by all three exit branches (STOP_LOSS exits get costs but **no** slippage — the tape already does that).

- [ ] **Step 1: Write the failing test**:

```python
def test_costs_and_slippage_reduce_pnl():
    # entry 1000, exit 1010 -> gross +10 pts; net == 10 - (1000*25/10000) - (1000*0.0005)
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** helper in the `Trade` construction path (one `_net` call inside each Trade-returning block).
- [ ] **Step 4: Run** — PASS, then full suite.
- [ ] **Step 5: Commit** `git commit -m "backtest: round-trip costs + gated-entry slippage; PnL now net"`

### Task 7: Rename the misleading "OI weight"

**Files:**
- Modify: `sector_momentum_desk/stock_scorer.py:47-87`, `sector_momentum_desk/models.py` (`StockCandidate.oi_weight`), `sector_momentum_desk/backtester.py:249-252`, `sector_momentum_desk/engine.py` (Telegram copy), `tests/test_sector_momentum.py` (`test_oi_weight_in_bounds`)
- Test: `tests/test_sector_momentum.py` (rename + rerun)

**Context from review:** the "OI weight" is raw volume — no Open Interest data anywhere in the desk. Anyone sizing positions later will believe OI analytics stand behind it.

**Interfaces:**
- Consumes: nothing new.
- Produces: `StockCandidate.volume_weight: float` and `score = ret * (1.0 + volume_weight)`; Telegram line `wei: {s.volume_weight:.3f}`; test renamed `test_volume_weight_in_bounds`.

- [ ] **Step 1: Write the renamed test** `test_volume_weight_in_bounds` (same body as existing OI test).
- [ ] **Step 2: Run** — FAIL on old name.
- [ ] **Step 3: Implement** rename across the five files (`grep -rn "oi_weight\|OI weight" sector_momentum_desk/ tests/` until zero hits).
- [ ] **Step 4: Run** — PASS, full suite.
- [ ] **Step 5: Commit** `git commit -m "momentum: oi_weight -> volume_weight (it was never OI)"`

### Explicitly deferred (not in this plan)

- Routing the live momentum desk through `DataProvider`/Kite instead of yfinance (review should-fix 9) — needs a Phase-1-style spike on `zerodha_data.py` historical intervals and its own plan; the parity fixes in Task 5 are valid for either source.
- Execution layer for momentum exits (review should-fix 10) — blocked on the desk consolidation above; add a `shortcut:` marker now in `runner.py` header: `# shortcut: Telegram-only exits, no order path — wire ZerodhaOrderExecutor-style layer before disabling dry_run`.
- The "83.3% historical win rate" claim string stays deleted only if touched; it lives in `options_engine.py:174` docstring — one-line delete folded into Task 3's commit.

---

## Task order & dependencies

- Phase A tasks are sequential (Task 2 needs Task 1 legs; Task 4 and 8 fold into Task 1's helper). Recommended run order: 1 → 2 → 3 → 8 → 4.
- Phase B tasks are independent of each other and of Phase A: run 5, then 6, then 7, any time.
- One full-suite `pytest` run after Phase A completes and one after Phase B; commits go as a single commit series on the current working tree, matching existing repo practice.
