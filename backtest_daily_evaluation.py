import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

from model import Kronos, KronosTokenizer, KronosPredictor
from options_engine import snap_strike

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

TARGETS = [
    {"symbol": "^NSEI", "name": "NIFTY 50 Index", "step": 50},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "step": 20},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "step": 50},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "step": 20}
]


def run_daily_backtest():
    print("=" * 105)
    print("  📈  KRONOS-BASE DAILY PREDICTION BACKTEST — LAST 1 MONTH OF MARKET DATA")
    print("  Evaluating Daily Direction (UP/DOWN) & Option Selling Safety for NIFTY, RELIANCE, TCS & HDFCBANK")
    print("=" * 105 + "\n")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Loading Kronos-base foundation model on {device}...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print("[INIT] Model ready for backtesting.\n")

    overall_results = []
    lookback = 40
    test_days_count = 18  # ~1 month of trading days

    for target in TARGETS:
        symbol = target["symbol"]
        name = target["name"]
        step = target["step"]

        print("=" * 105)
        print(f"  TARGET: {name} ({symbol})")
        print("=" * 105)
        print(f"{'TEST DATE':<11} | {'TODAY CLOSE':<11} | {'PRED CLOSE':<11} | {'ACTUAL CLOSE':<12} | {'PRED DIR':<8} | {'ACTUAL DIR':<10} | {'DIR MATCH':<9} | {'OPTION STRIKE':<14} | {'OPTION RESULT'}")
        print("-" * 105)

        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo", interval="1d").reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()

        total_bars = len(df)
        if total_bars < lookback + test_days_count:
            print(f"[WARN] Not enough bars ({total_bars}) for {symbol}. Skipping.")
            continue

        asset_results = []
        start_idx = total_bars - test_days_count - 1

        for i in range(start_idx, total_bars - 1):
            train_df = df.iloc[i - lookback + 1 : i + 1].reset_index(drop=True)
            today_close = float(train_df["close"].iloc[-1])
            today_date = train_df["timestamps"].iloc[-1].strftime("%Y-%m-%d")

            actual_next_bar = df.iloc[i + 1]
            next_date = actual_next_bar["timestamps"].strftime("%Y-%m-%d")
            actual_close = float(actual_next_bar["close"])
            actual_high = float(actual_next_bar["high"])
            actual_low = float(actual_next_bar["low"])

            x_df = train_df[["open", "high", "low", "close"]]
            x_timestamp = train_df["timestamps"]
            y_timestamp = pd.Series([actual_next_bar["timestamps"]])

            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=1,
                T=0.7,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )

            pred_close = float(pred_df["close"].iloc[-1])
            pred_high = float(pred_df["high"].iloc[-1])
            pred_low = float(pred_df["low"].iloc[-1])

            # Directional comparison
            pred_move_pct = ((pred_close - today_close) / today_close) * 100.0
            actual_move_pct = ((actual_close - today_close) / today_close) * 100.0

            pred_dir = "UP" if pred_move_pct >= 0 else "DOWN"
            actual_dir = "UP" if actual_move_pct >= 0 else "DOWN"
            dir_match = (pred_dir == actual_dir)

            # Option Selling simulation:
            # If Bullish -> Sell Put at or below predicted low
            # If Bearish -> Sell Call at or above predicted high
            if pred_dir == "UP":
                safe_put = snap_strike(min(pred_low, today_close * 0.995), step, direction="floor")
                opt_strike = f"SELL {safe_put} PE"
                opt_won = (actual_low > safe_put)
            else:
                safe_call = snap_strike(max(pred_high, today_close * 1.005), step, direction="ceil")
                opt_strike = f"SELL {safe_call} CE"
                opt_won = (actual_high < safe_call)

            dir_str = "WIN" if dir_match else "LOSS"
            opt_str = "PROFIT (OTM)" if opt_won else "BREACHED"

            print(f"{next_date:<11} | {today_close:<11.1f} | {pred_close:<11.1f} | {actual_close:<12.1f} | {pred_dir:<8} | {actual_dir} ({actual_move_pct:+.2f}%) | {dir_str:<9} | {opt_strike:<14} | {opt_str}")

            res = {
                "symbol": symbol,
                "name": name,
                "date": next_date,
                "dir_match": dir_match,
                "opt_won": opt_won,
                "actual_move_pct": actual_move_pct
            }
            asset_results.append(res)
            overall_results.append(res)

        # Asset summary
        asset_dir_wins = sum(1 for r in asset_results if r["dir_match"])
        asset_opt_wins = sum(1 for r in asset_results if r["opt_won"])
        print("-" * 105)
        print(f"  📊 {name} Scorecard: Directional Win Rate = {asset_dir_wins}/{len(asset_results)} ({(asset_dir_wins/len(asset_results))*100:.1f}%) | Option Selling Win Rate = {asset_opt_wins}/{len(asset_results)} ({(asset_opt_wins/len(asset_results))*100:.1f}%)\n")

    # Global Scorecard
    total_samples = len(overall_results)
    total_dir_wins = sum(1 for r in overall_results if r["dir_match"])
    total_opt_wins = sum(1 for r in overall_results if r["opt_won"])

    dir_acc = (total_dir_wins / total_samples) * 100.0 if total_samples > 0 else 0
    opt_acc = (total_opt_wins / total_samples) * 100.0 if total_samples > 0 else 0

    print("=" * 60)
    print("  🏆  OVERALL 1-MONTH MULTI-ASSET SCORECARD")
    print("=" * 60)
    print(f"  • Total Daily Predictions Tested  : {total_samples}")
    print(f"  • Direction Matches (UP/DOWN)     : {total_dir_wins} / {total_samples} ({dir_acc:.1f}%)")
    print(f"  • Option Selling Wins (OTM Decay) : {total_opt_wins} / {total_samples} ({opt_acc:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_daily_backtest()
