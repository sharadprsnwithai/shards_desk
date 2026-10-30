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

BASE_CAPITAL = 100000.0  # ₹1,00,000 margin pool
NIFTY_LOT_SIZE = 65
SENSEX_LOT_SIZE = 10


def run_sl50_tp80_backtest():
    print("=" * 125)
    print("  🎯  KRONOS INTRADAY OPTION SELLING — 50% STOP-LOSS & 80% PROFIT BOOKING BACKTEST")
    print("  Rule: Exit at +80% Premium Decay (Profit Target) OR -50% Premium Expansion (Disciplined SL)")
    print("  Schedule: Mon/Tue/Fri = NIFTY 50 | Wed/Thu = BSE SENSEX | Base Capital: Rs. 1,00,000")
    print("=" * 125 + "\n")

    # 1. Fetch 5m data with local cache
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
    print(f"      Loaded {len(trading_days)} trading days.\n")

    # 2. Load model
    print("[2/3] Loading Kronos-base foundation model...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    print("      Model loaded successfully.\n")

    # 3. Simulate candle-by-candle path
    print("[3/3] Simulating candle-by-candle 50% SL and 80% Profit Booking...\n")
    print(f"{'DATE':<11} | {'DAY':<3} | {'INDEX':<7} | {'10AM SPOT':<10} | {'STRATEGY':<18} | {'PREMIUM':<8} | {'EXIT TRIGGER':<15} | {'EXIT TIME':<10} | {'NET PTS':<8} | {'P&L (Rs.)':<10} | {'ACCUM P&L'}")
    print("-" * 125)

    lookback = 100
    accumulated_pnl = 0.0
    trade_logs = []

    for idx in range(2, len(trading_days)):
        test_day = trading_days[idx]
        weekday_idx = test_day.weekday()
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday_idx]

        if weekday_idx in (0, 1, 4):
            traded_index = "NIFTY"
            raw_df = nifty_df
            step = 50
            lot_size = NIFTY_LOT_SIZE
            init_premium = 32.0   # Average 10:00 AM OTM premium
            delta = 0.18           # OTM Delta
        else:
            traded_index = "SENSEX"
            raw_df = sensex_df
            step = 100
            lot_size = SENSEX_LOT_SIZE
            init_premium = 95.0
            delta = 0.18

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

        pred_move_pct = ((proj_close - spot_10am) / spot_10am) * 100.0

        if pred_move_pct >= 0.20:
            bias = "BULLISH"
            sold_strike = snap_strike(min(pred_low, spot_10am * 0.995), step, direction="floor")
            strat_label = f"Sell {sold_strike} PE"
            is_put = True
        elif pred_move_pct <= -0.20:
            bias = "BEARISH"
            sold_strike = snap_strike(max(pred_high, spot_10am * 1.005), step, direction="ceil")
            strat_label = f"Sell {sold_strike} CE"
            is_put = False
        else:
            bias = "NEUTRAL"
            put_strike = snap_strike(spot_10am * 0.994, step, direction="floor")
            call_strike = snap_strike(spot_10am * 1.006, step, direction="ceil")
            strat_label = f"{put_strike}P/{call_strike}C"
            is_put = None  # Strangle

        # Candle-by-candle simulation
        total_session_candles = len(rest_candles)
        exit_trigger = "3:15 PM EOD"
        exit_time = "15:15"
        net_pts = 0.0

        target_exit_price = init_premium * 0.20  # 80% profit target (+80% decay)
        sl_exit_price = init_premium * 1.50      # 50% stop loss (+50% premium expansion)

        for c_idx, (_, candle) in enumerate(rest_candles.iterrows()):
            candle_close = float(candle['close'])
            candle_time = candle['timestamps'].strftime("%H:%M")

            # Daily theta decay accelerates over the session
            decay_factor = 1.0 - (0.75 * (c_idx / total_session_candles))

            if is_put is True:
                adverse_move = max(0.0, spot_10am - candle_close)
                favorable_move = max(0.0, candle_close - spot_10am)
                curr_prem = (init_premium * decay_factor) + (adverse_move * delta) - (favorable_move * delta * 0.5)
            elif is_put is False:
                adverse_move = max(0.0, candle_close - spot_10am)
                favorable_move = max(0.0, spot_10am - candle_close)
                curr_prem = (init_premium * decay_factor) + (adverse_move * delta) - (favorable_move * delta * 0.5)
            else:
                dev = abs(candle_close - spot_10am)
                curr_prem = (init_premium * decay_factor) + (dev * delta * 0.8)

            curr_prem = max(1.0, curr_prem)

            # Check Stop Loss (50% premium increase)
            if curr_prem >= sl_exit_price:
                exit_trigger = "STOP-LOSS (-50%)"
                exit_time = candle_time
                net_pts = -(init_premium * 0.50)
                break

            # Check Profit Booking (80% premium decay)
            if curr_prem <= target_exit_price:
                exit_trigger = "TARGET (+80%)"
                exit_time = candle_time
                net_pts = (init_premium * 0.80)
                break

        # If neither hit, exit at 3:15 EOD with accrued decay
        if exit_trigger == "3:15 PM EOD":
            net_pts = init_premium - curr_prem

        day_pnl = net_pts * lot_size
        accumulated_pnl += day_pnl

        print(f"{str(test_day):<11} | {day_name:<3} | {traded_index:<7} | {spot_10am:<10.1f} | {strat_label:<18} | {init_premium:<8.1f} | {exit_trigger:<15} | {exit_time:<10} | {net_pts:>+6.1f}   | Rs. {day_pnl:>+7,.0f} | Rs. {accumulated_pnl:>+8,.0f}")

        trade_logs.append({
            "date": test_day,
            "day": day_name,
            "trigger": exit_trigger,
            "net_pts": net_pts,
            "pnl": day_pnl
        })

    # Summary Statistics
    total_trades = len(trade_logs)
    target_hits = sum(1 for t in trade_logs if "TARGET" in t["trigger"])
    sl_hits = sum(1 for t in trade_logs if "STOP-LOSS" in t["trigger"])
    eod_exits = sum(1 for t in trade_logs if "EOD" in t["trigger"])

    wins = sum(1 for t in trade_logs if t["pnl"] > 0)
    win_rate = (wins / total_trades) * 100.0

    total_gains = sum(t["pnl"] for t in trade_logs if t["pnl"] > 0)
    total_losses = sum(abs(t["pnl"]) for t in trade_logs if t["pnl"] < 0)
    profit_factor = (total_gains / total_losses) if total_losses > 0 else float("inf")

    roi_pct = (accumulated_pnl / BASE_CAPITAL) * 100.0

    print("-" * 125)
    print("\n" + "=" * 65)
    print("  🏆  1-MONTH INTRADAY OPTION SELLING SCORECARD (50% SL / 80% TP)")
    print("=" * 65)
    print(f"  • Base Trading Capital Margin Pool : Rs. {BASE_CAPITAL:,.0f}")
    print(f"  • Total Trading Sessions Traded    : {total_trades} days")
    print(f"  • 🎯 Target Booked (+80% Decay)    : {target_hits} sessions ({(target_hits/total_trades)*100:.1f}%)")
    print(f"  • 🛑 Stop-Loss Triggered (-50% Cut): {sl_hits} sessions ({(sl_hits/total_trades)*100:.1f}%)")
    print(f"  • ⏱️ EOD Normal Exits (3:15 PM)    : {eod_exits} sessions ({(eod_exits/total_trades)*100:.1f}%)")
    print(f"  • Overall Winning Rate             : {wins} / {total_trades} ({win_rate:.1f}%)")
    print(f"  • Gross Profits (+80% Hits + EOD)  : +Rs. {total_gains:,.0f}")
    print(f"  • Gross Losses (-50% SL Cuts)      : -Rs. {total_losses:,.0f}")
    print(f"  • Profit Factor (Gains / Losses)   : {profit_factor:.2f}")
    print("  " + "─" * 60)
    print(f"  ⭐ NET MONTHLY RETURN (P&L)        : +Rs. {accumulated_pnl:,.0f}")
    print(f"  ⭐ MONTHLY ROI (Return on Margin)  : +{roi_pct:.2f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_sl50_tp80_backtest()
