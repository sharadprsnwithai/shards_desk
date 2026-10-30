import sys
import time
from pathlib import Path
from datetime import datetime, time as dtime
import pandas as pd
import numpy as np
import yfinance as yf
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Ensure local imports work
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

from model import Kronos, KronosTokenizer, KronosPredictor
from options_engine import snap_strike, determine_intraday_option_strategy

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
STEP = 50  # Nifty 50 point strike steps


def run_nifty_1month_backtest():
    print("=" * 100)
    print("  📊  KRONOS-BASE INTRADAY 10:00 AM ENGINE — 1-MONTH NIFTY 50 BACKTEST")
    print("  Testing 5-Minute Candlestick Sequence at 10:00 AM IST vs Actual 3:30 PM Close")
    print("=" * 100 + "\n")

    # 1. Fetch live 1-month 5m data
    print("[1/3] Downloading last 1 month 5-minute candles for NIFTY 50 (^NSEI)...")
    ticker = yf.Ticker("^NSEI")
    raw_df = ticker.history(period="1mo", interval="5m").reset_index()
    if raw_df.empty or len(raw_df) < 200:
        print("[ERR] Insufficient 5m data returned from Yahoo Finance.")
        return

    raw_df['timestamps'] = pd.to_datetime(raw_df['Datetime']).dt.tz_localize(None)
    raw_df = raw_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    raw_df = raw_df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].dropna()

    unique_days = sorted(list(set(raw_df['timestamps'].dt.date)))
    print(f"      Total trading days captured: {len(unique_days)} days ({len(raw_df)} candles)\n")

    # 2. Load Kronos model
    print("[2/3] Loading Kronos-base (~400MB) foundation model into memory...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print(f"      Model loaded successfully on {device}.\n")

    # 3. Iterate over trading days (start from day index 2 to ensure lookback context)
    print("[3/3] Executing 10:00 AM backtest loop across trading sessions...\n")
    print(f"{'DATE':<11} | {'10AM SPOT':<10} | {'PRED 3:30':<10} | {'ACTUAL 3:30':<11} | {'PRED BIAS':<9} | {'ACTUAL DIR':<10} | {'DIR MATCH':<9} | {'OPTION STRIKE':<14} | {'OPTION RESULT'}")
    print("-" * 115)

    lookback = 100
    results = []

    for idx in range(2, len(unique_days)):
        test_day = unique_days[idx]
        
        # Split day data: up to 10:00 AM vs after 10:00 AM
        day_mask = (raw_df['timestamps'].dt.date == test_day)
        day_candles = raw_df[day_mask]
        
        morn_candles = day_candles[day_candles['timestamps'].dt.time <= dtime(10, 0)]
        rest_candles = day_candles[day_candles['timestamps'].dt.time > dtime(10, 0)]

        if len(morn_candles) < 5 or len(rest_candles) < 20:
            continue  # Skip incomplete sessions

        # Cutoff index at 10:00 AM
        cutoff_idx = morn_candles.index[-1]
        if cutoff_idx < lookback:
            continue

        # Lookback candles up to 10:00 AM
        x_df = raw_df.loc[cutoff_idx - lookback + 1:cutoff_idx, ['open', 'high', 'low', 'close']].reset_index(drop=True)
        x_timestamp = raw_df.loc[cutoff_idx - lookback + 1:cutoff_idx, 'timestamps'].reset_index(drop=True)

        pred_len = len(rest_candles)
        last_time = x_timestamp.iloc[-1]
        y_timestamp = pd.Series([last_time + pd.Timedelta(minutes=5 * i) for i in range(1, pred_len + 1)])

        try:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=0.7,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
        except Exception as e:
            continue

        spot_10am = float(x_df['close'].iloc[-1])
        proj_close = float(pred_df['close'].iloc[-1])
        pred_low = float(pred_df['low'].min())
        pred_high = float(pred_df['high'].max())

        actual_close = float(rest_candles['close'].iloc[-1])
        actual_day_low = float(rest_candles['low'].min())
        actual_day_high = float(rest_candles['high'].max())

        # Evaluate Direction
        pred_move_pct = ((proj_close - spot_10am) / spot_10am) * 100.0
        actual_move_pct = ((actual_close - spot_10am) / spot_10am) * 100.0

        if pred_move_pct >= 0.20:
            pred_bias = "BULLISH"
        elif pred_move_pct <= -0.20:
            pred_bias = "BEARISH"
        else:
            pred_bias = "NEUTRAL"

        actual_dir = "UP" if actual_move_pct > 0 else "DOWN"
        
        # Directional win: Predicted Bullish and went UP, or Predicted Bearish and went DOWN
        # (Neutral counts as match if move was minor, under 0.25%)
        if pred_bias == "BULLISH":
            dir_match = (actual_move_pct >= 0)
        elif pred_bias == "BEARISH":
            dir_match = (actual_move_pct <= 0)
        else:
            dir_match = (abs(actual_move_pct) < 0.35)

        # Evaluate Option Selling Strategy
        if pred_bias == "BULLISH":
            # Sell OTM Put below support (safe floor)
            sold_strike = snap_strike(min(pred_low, spot_10am * 0.995), STEP, direction="floor")
            opt_type = "PE"
            # Won if actual low stayed strictly above sold Put
            opt_won = (actual_day_low > sold_strike)
            opt_label = f"SELL {sold_strike} PE"
        elif pred_bias == "BEARISH":
            # Sell OTM Call above resistance (safe ceiling)
            sold_strike = snap_strike(max(pred_high, spot_10am * 1.005), STEP, direction="ceil")
            opt_type = "CE"
            # Won if actual high stayed strictly below sold Call
            opt_won = (actual_day_high < sold_strike)
            opt_label = f"SELL {sold_strike} CE"
        else:
            # Neutral: Short Strangle (Sell Put & Call)
            put_strike = snap_strike(spot_10am * 0.994, STEP, direction="floor")
            call_strike = snap_strike(spot_10am * 1.006, STEP, direction="ceil")
            opt_won = (actual_day_low > put_strike and actual_day_high < call_strike)
            opt_label = f"{put_strike}P/{call_strike}C"

        dir_str = "WIN" if dir_match else "LOSS"
        opt_str = "PROFIT (OTM)" if opt_won else "BREACHED"

        print(f"{str(test_day):<11} | {spot_10am:<10.1f} | {proj_close:<10.1f} | {actual_close:<11.1f} | {pred_bias:<9} | {actual_dir} ({actual_move_pct:+.2f}%) | {dir_str:<9} | {opt_label:<14} | {opt_str}")

        results.append({
            "date": test_day,
            "spot_10am": spot_10am,
            "proj_close": proj_close,
            "actual_close": actual_close,
            "pred_bias": pred_bias,
            "actual_move_pct": actual_move_pct,
            "dir_match": dir_match,
            "opt_won": opt_won
        })

    # 4. Summary Scorecard
    total_evaluated = len(results)
    if total_evaluated == 0:
        print("[WARN] No sessions evaluated.")
        return

    dir_wins = sum(1 for r in results if r["dir_match"])
    opt_wins = sum(1 for r in results if r["opt_won"])

    dir_accuracy = (dir_wins / total_evaluated) * 100.0
    opt_winrate = (opt_wins / total_evaluated) * 100.0

    print("-" * 115)
    print("\n" + "=" * 50)
    print("  🏆  1-MONTH INTRADAY SCORECARD SUMMARY")
    print("=" * 50)
    print(f"  • Total Trading Sessions Evaluated : {total_evaluated} days")
    print(f"  • Directional Correct Matches      : {dir_wins} / {total_evaluated}")
    print(f"  • Directional Predictive Accuracy  : {dir_accuracy:.1f}%")
    print(f"  • Option Selling Trades Won (OTM)  : {opt_wins} / {total_evaluated}")
    print(f"  • Option Selling Win Rate (10AM)   : {opt_winrate:.1f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_nifty_1month_backtest()
