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

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

TARGETS = [
    {"symbol": "^NSEI", "name": "NIFTY 50 Index"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd"}
]


def run_monthly_2year_backtest():
    print("=" * 115)
    print("  📅  KRONOS-BASE 2-YEAR MONTHLY (30-DAY) POSITIONAL FORECAST BACKTEST")
    print("  Testing 24 Monthly Cycles (2024 to 2026) Across NIFTY, RELIANCE, TCS & HDFCBANK")
    print("=" * 115 + "\n")

    # Ensure cache directory exists
    cache_dir = CUR_DIR / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download / Load 3-year daily data
    print("[1/3] Loading 3-year historical daily candlestick data...")
    market_data = {}
    for target in TARGETS:
        sym = target["symbol"]
        cache_file = cache_dir / f"{sym.replace('^', '').replace('.NS', '')}_daily_3y.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        else:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="3y", interval="1d").reset_index()
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
            df.to_parquet(cache_file, index=False)
        market_data[sym] = df
        print(f"      {target['name']:<25}: {len(df)} daily candles loaded.")

    # 2. Load Kronos model
    print("\n[2/3] Loading Kronos-base foundation model into memory...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print(f"      Model loaded successfully on {device}.\n")

    # 3. Step through 24 monthly periods (~21 trading days each)
    print("[3/3] Executing 2-year sliding monthly evaluation...\n")

    lookback = 90
    forecast_step = 21  # 21 trading days = ~1 calendar month
    total_months_to_test = 24

    all_results = []

    for target in TARGETS:
        sym = target["symbol"]
        name = target["name"]
        df = market_data[sym]

        print("=" * 115)
        print(f"  ASSET: {name} ({sym}) — 24 MONTHLY PERIODS")
        print("=" * 115)
        print(f"{'MONTH':<3} | {'START DATE':<11} | {'END DATE':<11} | {'START SPOT':<11} | {'PRED CLOSE':<11} | {'ACTUAL CLOSE':<12} | {'PRED DIR':<8} | {'ACTUAL DIR':<10} | {'DIR MATCH':<9} | {'RANGE BOUNDS':<23} | {'RANGE WIN'}")
        print("-" * 115)

        total_bars = len(df)
        asset_records = []

        # Find starting index for 24 months back
        # 24 periods * 21 days = ~504 trading days
        start_eval_idx = total_bars - (total_months_to_test * forecast_step)

        for m_idx in range(total_months_to_test):
            curr_idx = start_eval_idx + (m_idx * forecast_step)
            next_idx = curr_idx + forecast_step

            if next_idx >= total_bars:
                break

            train_df = df.iloc[curr_idx - lookback : curr_idx].reset_index(drop=True)
            future_df = df.iloc[curr_idx : next_idx].reset_index(drop=True)

            start_spot = float(train_df["close"].iloc[-1])
            start_date_str = train_df["timestamps"].iloc[-1].strftime("%Y-%m-%d")
            end_date_str = future_df["timestamps"].iloc[-1].strftime("%Y-%m-%d")

            x_df = train_df[["open", "high", "low", "close"]]
            x_ts = train_df["timestamps"]
            y_ts = future_df["timestamps"]

            pred_len = len(future_df)

            try:
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=pred_len,
                    T=0.75,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False
                )
            except Exception:
                continue

            pred_close = float(pred_df["close"].iloc[-1])
            pred_support = float(pred_df["low"].min())
            pred_resistance = float(pred_df["high"].max())

            actual_close = float(future_df["close"].iloc[-1])
            actual_min_low = float(future_df["low"].min())
            actual_max_high = float(future_df["high"].max())

            # Directional Evaluation
            pred_move_pct = ((pred_close - start_spot) / start_spot) * 100.0
            actual_move_pct = ((actual_close - start_spot) / start_spot) * 100.0

            pred_dir = "UP" if pred_move_pct >= 0 else "DOWN"
            actual_dir = "UP" if actual_move_pct >= 0 else "DOWN"
            dir_match = (pred_dir == actual_dir)

            # Range Containment Evaluation (with 1.5% buffer for market volatility)
            range_contained = (actual_min_low >= pred_support * 0.985) and (actual_max_high <= pred_resistance * 1.015)

            dir_str = "YES" if dir_match else "NO"
            range_str = "SAFE" if range_contained else "BREACHED"
            range_bounds_str = f"[{pred_support:,.0f} - {pred_resistance:,.0f}]"

            print(f"{m_idx+1:<3} | {start_date_str:<11} | {end_date_str:<11} | {start_spot:<11.1f} | {pred_close:<11.1f} | {actual_close:<12.1f} | {pred_dir:<8} | {actual_dir} ({actual_move_pct:+.1f}%)  | {dir_str:<9} | {range_bounds_str:<23} | {range_str}")

            rec = {
                "symbol": sym,
                "name": name,
                "month_num": m_idx + 1,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "dir_match": dir_match,
                "range_contained": range_contained
            }
            asset_records.append(rec)
            all_results.append(rec)

        # Asset summary
        a_dir_wins = sum(1 for r in asset_records if r["dir_match"])
        a_range_wins = sum(1 for r in asset_records if r["range_contained"])
        a_total = len(asset_records)
        print("-" * 115)
        print(f"  📊 {name} 2-Year Summary: Directional Matches = {a_dir_wins}/{a_total} ({(a_dir_wins/a_total)*100:.1f}%) | Range Containment = {a_range_wins}/{a_total} ({(a_range_wins/a_total)*100:.1f}%)\n")

    # Global Scorecard
    total_evals = len(all_results)
    total_dir_wins = sum(1 for r in all_results if r["dir_match"])
    total_range_wins = sum(1 for r in all_results if r["range_contained"])

    dir_rate = (total_dir_wins / total_evals) * 100.0 if total_evals > 0 else 0
    range_rate = (total_range_wins / total_evals) * 100.0 if total_evals > 0 else 0

    print("=" * 70)
    print("  🏆  OVERALL 2-YEAR MONTHLY SCORECARD (ALL 4 ASSETS)")
    print("=" * 70)
    print(f"  • Total 30-Day Monthly Forecasts Evaluated : {total_evals}")
    print(f"  • Direction Matches (UP/DOWN Correct)      : {total_dir_wins} / {total_evals} ({dir_rate:.1f}%)")
    print(f"  • Range Containment (Option Selling Safe)  : {total_range_wins} / {total_evals} ({range_rate:.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_monthly_2year_backtest()
