import sys
from pathlib import Path
from datetime import datetime, time as dtime
import pandas as pd
import numpy as np

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

BASE_CAPITAL = 100000.0  # ₹1,00,000 margin
NIFTY_LOT = 65
SENSEX_LOT = 10


def run_comparative_backtest():
    print("=" * 125)
    print("  🔬  COMPARATIVE BACKTEST: STRATEGY 1 (4-OTM STRIKES) vs STRATEGY 2 (NEXT-EXPIRY STRIKES)")
    print("  Testing 18 Trading Sessions on 5-Minute Historical Candlesticks | Base Capital: Rs. 1,00,000")
    print("=" * 125 + "\n")

    # Load cached data
    nifty_df = pd.read_parquet(CUR_DIR / "data_cache/nifty_5m_1mo.parquet")
    nifty_df['timestamps'] = pd.to_datetime(nifty_df['timestamps'])

    sensex_df = pd.read_parquet(CUR_DIR / "data_cache/sensex_5m_1mo.parquet")
    sensex_df['timestamps'] = pd.to_datetime(sensex_df['timestamps'])

    trading_days = sorted(list(set(nifty_df['timestamps'].dt.date)))

    # Load Kronos
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    lookback = 100

    strat1_results = []
    strat2_results = []

    print(f"{'DATE':<11} | {'DAY':<3} | {'INDEX':<6} | {'10AM SPOT':<10} | {'PRED BIAS':<9} | {'STRAT 1: 4-OTM STRIKE':<25} | {'RES 1':<8} | {'STRAT 2: NEXT-EXPIRY':<24} | {'RES 2'}")
    print("-" * 125)

    for idx in range(2, len(trading_days)):
        test_day = trading_days[idx]
        weekday_idx = test_day.weekday()
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday_idx]

        if weekday_idx in (0, 1, 4):
            traded_index = "NIFTY"
            raw_df = nifty_df
            step = 50
            lot_size = NIFTY_LOT
            otm_distance = 4 * step   # 200 pts OTM
            s1_premium = 15.0         # 4-OTM premium ~15 pts
            s2_premium = 68.0         # Next-expiry 2-OTM premium ~68 pts
            s2_theta_pts = 10.5       # Next-expiry daily theta decay (~15%)
            s2_sl_pts = 25.0          # Stop loss on next-expiry
        else:
            traded_index = "SENSEX"
            raw_df = sensex_df
            step = 100
            lot_size = SENSEX_LOT
            otm_distance = 4 * step   # 400 pts OTM
            s1_premium = 48.0         # 4-OTM premium ~48 pts
            s2_premium = 210.0        # Next-expiry 2-OTM premium ~210 pts
            s2_theta_pts = 32.0       # Next-expiry daily theta decay (~15%)
            s2_sl_pts = 75.0          # Stop loss on next-expiry

        day_candles = raw_df[raw_df['timestamps'].dt.date == test_day]
        morn = day_candles[day_candles['timestamps'].dt.time <= dtime(10, 0)]
        rest = day_candles[day_candles['timestamps'].dt.time > dtime(10, 0)]

        if len(morn) < 5 or len(rest) < 20:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < lookback:
            continue

        x_df = raw_df.loc[cutoff_idx - lookback + 1:cutoff_idx, ['open', 'high', 'low', 'close']].reset_index(drop=True)
        x_ts = raw_df.loc[cutoff_idx - lookback + 1:cutoff_idx, 'timestamps'].reset_index(drop=True)

        pred_len = len(rest)
        y_ts = pd.Series([x_ts.iloc[-1] + pd.Timedelta(minutes=5 * i) for i in range(1, pred_len + 1)])

        pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len, T=0.7, top_p=0.9, sample_count=1, verbose=False)

        spot_10am = float(x_df['close'].iloc[-1])
        proj_close = float(pred_df['close'].iloc[-1])

        actual_close = float(rest['close'].iloc[-1])
        actual_min_low = float(rest['low'].min())
        actual_max_high = float(rest['high'].max())

        pred_move_pct = ((proj_close - spot_10am) / spot_10am) * 100.0
        bias = "BULLISH" if pred_move_pct >= 0.20 else ("BEARISH" if pred_move_pct <= -0.20 else "NEUTRAL")

        # ─────────────────────────────────────────────────────────────
        # STRATEGY 1: 4 OTM Strike Selling (Current Expiry)
        # ─────────────────────────────────────────────────────────────
        if bias == "BULLISH":
            s1_strike = snap_strike(spot_10am - otm_distance, step, direction="floor")
            s1_label = f"Sell {s1_strike} PE (-{otm_distance}pts)"
            # Won if actual low stayed strictly above 4-OTM put
            s1_won = (actual_min_low > s1_strike)
        elif bias == "BEARISH":
            s1_strike = snap_strike(spot_10am + otm_distance, step, direction="ceil")
            s1_label = f"Sell {s1_strike} CE (+{otm_distance}pts)"
            # Won if actual high stayed strictly below 4-OTM call
            s1_won = (actual_max_high < s1_strike)
        else:
            put_strike = snap_strike(spot_10am - otm_distance, step, direction="floor")
            call_strike = snap_strike(spot_10am + otm_distance, step, direction="ceil")
            s1_label = f"{put_strike}P/{call_strike}C (-+4OTM)"
            s1_won = (actual_min_low > put_strike and actual_max_high < call_strike)

        # S1 P&L: If unbreached, captures full 4-OTM decay (+15 pts Nifty / +48 pts Sensex)
        # If breached, hits standard stop loss (-1.5x premium)
        if s1_won:
            s1_pnl = s1_premium * lot_size
            s1_str = "WIN"
        else:
            s1_pnl = -(s1_premium * 1.5) * lot_size
            s1_str = "BREACH"

        strat1_results.append({"date": test_day, "won": s1_won, "pnl": s1_pnl})

        # ─────────────────────────────────────────────────────────────
        # STRATEGY 2: Next Weekly Expiry Strike Selling (DTE 6-10 days)
        # ─────────────────────────────────────────────────────────────
        # Next expiry has low gamma and smooth delta (~0.12).
        # We sell 2-OTM strikes in next weekly expiry.
        s2_dist = 2 * step
        if bias == "BULLISH":
            s2_strike = snap_strike(spot_10am - s2_dist, step, direction="floor")
            s2_label = f"NextWk {s2_strike} PE"
            # Breached only if actual low dropped severely past 2-OTM strike (> 1.2%)
            s2_won = (actual_min_low > s2_strike - (step * 0.5))
        elif bias == "BEARISH":
            s2_strike = snap_strike(spot_10am + s2_dist, step, direction="ceil")
            s2_label = f"NextWk {s2_strike} CE"
            s2_won = (actual_max_high < s2_strike + (step * 0.5))
        else:
            p_st = snap_strike(spot_10am - s2_dist, step, direction="floor")
            c_st = snap_strike(spot_10am + s2_dist, step, direction="ceil")
            s2_label = f"NextWk {p_st}P/{c_st}C"
            s2_won = (actual_min_low > p_st - (step * 0.5) and actual_max_high < c_st + (step * 0.5))

        # S2 P&L: In next expiry, if unbreached, captures steady intraday theta decay
        if s2_won:
            s2_pnl = s2_theta_pts * lot_size
            s2_str = "WIN"
        else:
            s2_pnl = -s2_sl_pts * lot_size
            s2_str = "SL HIT"

        strat2_results.append({"date": test_day, "won": s2_won, "pnl": s2_pnl})

        print(f"{str(test_day):<11} | {day_name:<3} | {traded_index:<6} | {spot_10am:<10.1f} | {bias:<9} | {s1_label:<25} | {s1_str:<8} | {s2_label:<24} | {s2_str}")

    print("-" * 125)

    # ─────────────────────────────────────────────────────────────
    # SUMMARY SCORECARDS COMPARISON
    # ─────────────────────────────────────────────────────────────
    total = len(strat1_results)

    # Strategy 1 Stats
    s1_wins = sum(1 for r in strat1_results if r["won"])
    s1_winrate = (s1_wins / total) * 100.0
    s1_net_pnl = sum(r["pnl"] for r in strat1_results)
    s1_roi = (s1_net_pnl / BASE_CAPITAL) * 100.0
    s1_gains = sum(r["pnl"] for r in strat1_results if r["pnl"] > 0)
    s1_losses = sum(abs(r["pnl"]) for r in strat1_results if r["pnl"] < 0)
    s1_pf = (s1_gains / s1_losses) if s1_losses > 0 else float("inf")

    # Strategy 2 Stats
    s2_wins = sum(1 for r in strat2_results if r["won"])
    s2_winrate = (s2_wins / total) * 100.0
    s2_net_pnl = sum(r["pnl"] for r in strat2_results)
    s2_roi = (s2_net_pnl / BASE_CAPITAL) * 100.0
    s2_gains = sum(r["pnl"] for r in strat2_results if r["pnl"] > 0)
    s2_losses = sum(abs(r["pnl"]) for r in strat2_results if r["pnl"] < 0)
    s2_pf = (s2_gains / s2_losses) if s2_losses > 0 else float("inf")

    print("\n" + "=" * 90)
    print("  🏆  FINAL 1-MONTH COMPARATIVE RESULTS: 4-OTM vs NEXT-EXPIRY")
    print("=" * 90)
    print(f"{'METRIC':<35} | {'STRATEGY 1 (4-OTM)':<25} | {'STRATEGY 2 (NEXT-EXPIRY)'}")
    print("-" * 90)
    print(f"{'Sessions Traded':<35} | {total:<25} | {total}")
    print(f"{'Winning Sessions':<35} | {s1_wins} / {total} ({s1_winrate:.1f}%)        | {s2_wins} / {total} ({s2_winrate:.1f}%)")
    print(f"{'Losing Sessions':<35} | {total - s1_wins} / {total} ({100 - s1_winrate:.1f}%)        | {total - s2_wins} / {total} ({100 - s2_winrate:.1f}%)")
    print(f"{'Gross Profit Generated':<35} | +Rs. {s1_gains:<20,.0f} | +Rs. {s2_gains:,.0f}")
    print(f"{'Gross Losses':<35} | -Rs. {s1_losses:<20,.0f} | -Rs. {s2_losses:,.0f}")
    print(f"{'Profit Factor':<35} | {s1_pf:<25.2f} | {s2_pf:.2f}")
    print("─" * 90)
    print(f"{'⭐ NET MONTHLY P&L':<35} | +Rs. {s1_net_pnl:<20,.0f} | +Rs. {s2_net_pnl:,.0f}")
    print(f"{'⭐ MONTHLY ROI (% on Rs 1L)':<35} | +{s1_roi:.2f}%{'':<20} | +{s2_roi:.2f}%")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_comparative_backtest()
