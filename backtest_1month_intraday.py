import sys
from pathlib import Path
from datetime import time as dtime
import pandas as pd
import yfinance as yf

# Ensure local imports work
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

from nifty_direction_engine.signal_engine import SignalEngine

def run_1month_backtest():
    print(f"Running 1-month Nifty Intraday Backtest...")

    ticker = yf.Ticker("^NSEI")
    raw_df = ticker.history(period="30d", interval="5m").reset_index()
    
    raw_df['timestamps'] = pd.to_datetime(raw_df['Datetime']).dt.tz_localize(None)
    raw_df = raw_df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    raw_df = raw_df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].dropna()

    unique_days = sorted(list(set(raw_df['timestamps'].dt.date)))
    engine = SignalEngine(use_mock=True)

    total_signals = 0
    hits = 0
    rejected = 0

    print(f"| {'DATE':<12} | {'BIAS':<10} | {'RESULT':<10} |")
    print("-" * 40)

    for test_day in unique_days:
        day_candles = raw_df[raw_df['timestamps'].dt.date == test_day]
        morn_candles = day_candles[day_candles['timestamps'].dt.time <= dtime(10, 0)]
        target_window = day_candles[(day_candles['timestamps'].dt.time > dtime(10, 0)) & (day_candles['timestamps'].dt.time <= dtime(11, 30))]

        if len(morn_candles) < 5 or len(target_window) < 5: continue

        # Simulate OI Flow (Mocking)
        spot_10am = float(morn_candles['close'].iloc[-1])
        atm_strike = int(round(spot_10am / 50) * 50)
        
        # Simple dummy chains
        chain_0915 = pd.DataFrame({'strike': [atm_strike], 'type': ['PE'], 'oi': [1000]})
        chain_1000 = pd.DataFrame({'strike': [atm_strike], 'type': ['PE'], 'oi': [1200]})

        signal_res = engine.generate_signal(morn_candles, chain_0915, chain_1000, atm_strike)
        
        bias = signal_res.get("final_bias")
        if bias == "SIDEWAYS_NO_TRADE":
            rejected += 1
            print(f"| {str(test_day):<12} | {bias:<10} | -          |")
            continue

        total_signals += 1
        actual_ret = (float(target_window['close'].iloc[-1]) - spot_10am) / spot_10am
        is_hit = (bias == "BULLISH" and actual_ret > 0.0005) or (bias == "BEARISH" and actual_ret < -0.0005)
        
        if is_hit: hits += 1
        print(f"| {str(test_day):<12} | {bias:<10} | {'HIT' if is_hit else 'MISS'}    |")

    print(f"\nWin Rate: {hits/max(1, total_signals)*100:.1f}% ({hits}/{total_signals}) | Rejected: {rejected}")

if __name__ == "__main__":
    run_1month_backtest()
