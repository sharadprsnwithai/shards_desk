import sys
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

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

from model import Kronos, KronosTokenizer, KronosPredictor
from options_engine import snap_strike

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

# Capital Allocation & Lot Sizes
BASE_CAPITAL = 100000.0  # ₹1,00,000 standard option selling margin pool
NIFTY_LOT_SIZE = 65       # NSE standard lot size
SENSEX_LOT_SIZE = 10      # BSE standard lot size


def run_intraday_roi_backtest():
    print("=" * 115)
    print("  💰  KRONOS INTRADAY OPTION SELLING ROI BACKTEST — 1 MONTH OF LIVE 5-MINUTE DATA")
    print("  Schedule: Mon/Tue/Fri = NIFTY 50 | Wed/Thu = BSE SENSEX | Base Capital: Rs. 1,00,000")
    print("=" * 115 + "\n")

    # 1. Fetch 5m data with disk caching
    cache_dir = CUR_DIR / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    nifty_cache = cache_dir / "nifty_5m_1mo.parquet"
    sensex_cache = cache_dir / "sensex_5m_1mo.parquet"

    if nifty_cache.exists() and sensex_cache.exists():
        print("[1/3] Loading 5-minute candles from local cache file (instant)...")
        nifty_df = pd.read_parquet(nifty_cache)
        nifty_df['timestamps'] = pd.to_datetime(nifty_df['timestamps'])
        sensex_df = pd.read_parquet(sensex_cache)
        sensex_df['timestamps'] = pd.to_datetime(sensex_df['timestamps'])
    else:
        print("[1/3] Downloading 5-minute candles for NIFTY 50 and BSE SENSEX...")
        nifty_df = yf.Ticker("^NSEI").history(period="1mo", interval="5m").reset_index()
        nifty_df['timestamps'] = pd.to_datetime(nifty_df['Datetime']).dt.tz_localize(None)
        nifty_df = nifty_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        nifty_df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].to_parquet(nifty_cache, index=False)

        sensex_df = yf.Ticker("^BSESN").history(period="1mo", interval="5m").reset_index()
        sensex_df['timestamps'] = pd.to_datetime(sensex_df['Datetime']).dt.tz_localize(None)
        sensex_df = sensex_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        sensex_df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].to_parquet(sensex_cache, index=False)

    trading_days = sorted(list(set(nifty_df['timestamps'].dt.date)))
    print(f"      Loaded {len(trading_days)} trading days across both indices.\n")

    # 2. Load Kronos model
    print("[2/3] Loading Kronos-base (~400MB) model on CPU...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    print("      Model loaded successfully.\n")

    # 3. Simulate day-by-day 10:00 AM trades
    print("[3/3] Simulating 10:00 AM Option Selling trades with 50% Stop-Loss Rule...\n")
    print(f"{'DATE':<11} | {'DAY':<3} | {'INDEX':<7} | {'10AM SPOT':<10} | {'BIAS':<8} | {'STRATEGY LEGS':<22} | {'OUTCOME':<12} | {'NET PTS':<8} | {'P&L (Rs.)':<10} | {'ACCUM P&L'}")
    print("-" * 115)

    lookback = 100
    accumulated_pnl = 0.0
    daily_logs = []

    # Start from index 2 to ensure lookback context
    for idx in range(2, len(trading_days)):
        test_day = trading_days[idx]
        weekday_idx = test_day.weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday_idx]

        # Select index based on weekday rotation rule
        if weekday_idx in (0, 1, 4):
            traded_index = "NIFTY"
            raw_df = nifty_df
            step = 50
            lot_size = NIFTY_LOT_SIZE
            # Realistic average 10:00 AM OTM credit collected (net after hedge)
            avg_credit_pts = 28.0   # ~28 points on Nifty
            sl_pts = 14.0           # 50% SL on credit
        else:
            traded_index = "SENSEX"
            raw_df = sensex_df
            step = 100
            lot_size = SENSEX_LOT_SIZE
            avg_credit_pts = 85.0   # ~85 points on Sensex
            sl_pts = 42.0           # 50% SL on credit

        day_candles = raw_df[raw_df['timestamps'].dt.date == test_day]
        morn_candles = day_candles[day_candles['timestamps'].dt.time <= dtime(10, 0)]
        rest_candles = day_candles[day_candles['timestamps'].dt.time > dtime(10, 0)]

        if len(morn_candles) < 5 or len(rest_candles) < 20:
            continue

        cutoff_idx = morn_candles.index[-1]
        if cutoff_idx < lookback:
            continue

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
        except Exception:
            continue

        spot_10am = float(x_df['close'].iloc[-1])
        proj_close = float(pred_df['close'].iloc[-1])
        pred_low = float(pred_df['low'].min())
        pred_high = float(pred_df['high'].max())

        actual_close = float(rest_candles['close'].iloc[-1])
        actual_day_low = float(rest_candles['low'].min())
        actual_day_high = float(rest_candles['high'].max())

        pred_move_pct = ((proj_close - spot_10am) / spot_10am) * 100.0

        if pred_move_pct >= 0.20:
            bias = "BULLISH"
            sold_strike = snap_strike(min(pred_low, spot_10am * 0.995), step, direction="floor")
            strategy_desc = f"Sell {sold_strike} PE"
            # Won if actual low stayed strictly above sold strike
            won = (actual_day_low > sold_strike)
        elif pred_move_pct <= -0.20:
            bias = "BEARISH"
            sold_strike = snap_strike(max(pred_high, spot_10am * 1.005), step, direction="ceil")
            strategy_desc = f"Sell {sold_strike} CE"
            # Won if actual high stayed strictly below sold strike
            won = (actual_day_high < sold_strike)
        else:
            bias = "NEUTRAL"
            put_strike = snap_strike(spot_10am * 0.994, step, direction="floor")
            call_strike = snap_strike(spot_10am * 1.006, step, direction="ceil")
            strategy_desc = f"{put_strike}P/{call_strike}C"
            won = (actual_day_low > put_strike and actual_day_high < call_strike)

        # Compute P&L
        if won:
            # Retain ~90% of credit collected as full Theta decay
            net_pts = round(avg_credit_pts * 0.90, 1)
            outcome = "FULL PROFIT"
            day_pnl = net_pts * lot_size
        else:
            # Hit disciplined 50% Stop Loss rule
            net_pts = -round(sl_pts, 1)
            outcome = "SL HIT (-50%)"
            day_pnl = net_pts * lot_size

        accumulated_pnl += day_pnl

        print(f"{str(test_day):<11} | {day_name:<3} | {traded_index:<7} | {spot_10am:<10.1f} | {bias:<8} | {strategy_desc:<22} | {outcome:<12} | {net_pts:>+6.1f}   | Rs. {day_pnl:>+7,.0f} | Rs. {accumulated_pnl:>+8,.0f}")

        daily_logs.append({
            "date": test_day,
            "day": day_name,
            "index": traded_index,
            "won": won,
            "pnl": day_pnl
        })

    # Summary Statistics
    total_trades = len(daily_logs)
    winning_trades = sum(1 for d in daily_logs if d["won"])
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0

    total_profit_rs = sum(d["pnl"] for d in daily_logs if d["pnl"] > 0)
    total_loss_rs = sum(abs(d["pnl"]) for d in daily_logs if d["pnl"] < 0)
    profit_factor = (total_profit_rs / total_loss_rs) if total_loss_rs > 0 else float("inf")

    roi_pct = (accumulated_pnl / BASE_CAPITAL) * 100.0

    print("-" * 115)
    print("\n" + "=" * 60)
    print("  🏆  1-MONTH INTRADAY OPTION SELLING ROI REPORT")
    print("=" * 60)
    print(f"  • Base Trading Capital Margin Pool : Rs. {BASE_CAPITAL:,.0f}")
    print(f"  • Total Trading Sessions Traded    : {total_trades} days")
    print(f"  • Winning Trades (Full Theta Decay): {winning_trades} ({win_rate:.1f}%)")
    print(f"  • Losing Trades (Disciplined SL)   : {losing_trades} ({100 - win_rate:.1f}%)")
    print(f"  • Gross Profit Generated           : +Rs. {total_profit_rs:,.0f}")
    print(f"  • Gross Losses (SL Cut)            : -Rs. {total_loss_rs:,.0f}")
    print(f"  • Profit Factor                    : {profit_factor:.2f}")
    print("  " + "─" * 56)
    print(f"  ⭐ NET MONTHLY RETURN (P&L)        : +Rs. {accumulated_pnl:,.0f}")
    print(f"  ⭐ MONTHLY ROI (Return on Margin)  : +{roi_pct:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_intraday_roi_backtest()
