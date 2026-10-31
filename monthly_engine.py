import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import torch

from data_provider import DataProvider
from options_engine import calculate_range_probability, determine_option_strategy
from telegram_notifier import send_telegram_message, format_options_report
from monthly_direction_engine.monthly_lstm_model import MonthlyLSTMPredictor

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

logger = logging.getLogger(__name__)

# Search paths for kronos_lib
CUR_DIR = Path(__file__).resolve().parent
PARENT_DIR = CUR_DIR.parent
for p in [CUR_DIR / "kronos_lib", PARENT_DIR / "kronos_lib"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
LSTM_WEIGHTS_PATH = CUR_DIR / "monthly_direction_engine" / "monthly_lstm_weights.pt"

# Core 6 Assets: Benchmarks & High-Volume Equities
# Lot sizes aligned with current exchange / Zerodha contracts
MONTHLY_ASSETS = [
    {"symbol": "^NSEI", "name": "NIFTY 50", "step": 50, "wing": 300, "lot_size": 65},
    {"symbol": "^NSEBANK", "name": "BANK NIFTY", "step": 100, "wing": 500, "lot_size": 30},
    {"symbol": "^BSESN", "name": "SENSEX", "step": 100, "wing": 600, "lot_size": 20},
    {"symbol": "RELIANCE.NS", "name": "RELIANCE", "step": 20, "wing": 100, "lot_size": 500},
    {"symbol": "TCS.NS", "name": "TCS", "step": 20, "wing": 100, "lot_size": 225},
    {"symbol": "HDFCBANK.NS", "name": "HDFCBANK", "step": 10, "wing": 50, "lot_size": 650},
]


class FallbackPredictor:
    """Removed — synthetic predictions must not reach order paths.
    Kept as a tombstone so git history shows the removal clearly."""
    pass


def get_kronos_predictor():
    """Initializes and caches the Kronos foundation model predictor."""
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"[MonthlyEngine] Loading Kronos foundation model on device: {device}...")
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
        return KronosPredictor(model, tokenizer, device=device, max_context=512)
    except Exception as e:
        logger.warning(f"[MonthlyEngine] Could not load Kronos foundation model: {e}. Using simulated quant forecaster.")
        return FallbackPredictor()


def run_monthly_analysis(send_telegram: bool = True) -> List[Dict[str, Any]]:
    """
    Executes the 30-day quantitative options selling analysis for target assets.
    """
    print("\n" + "=" * 90)
    print("  🚀  SHARAD'S 30-DAY MONTHLY OPTION SELLING DESK (QUANTITATIVE AI ENGINE)")
    print("  Zero LLM / No Ollama | Powered by Kronos-base (~400MB) & NSE Tick Architecture")
    print("=" * 90 + "\n")

    data_provider = DataProvider()
    predictor = get_kronos_predictor()
    lstm_predictor = MonthlyLSTMPredictor(
        weights_path=LSTM_WEIGHTS_PATH if LSTM_WEIGHTS_PATH.exists() else None
    )

    lookback = 120
    pred_len = 30
    forecast_results = []

    for asset in MONTHLY_ASSETS:
        symbol = asset["symbol"]
        name = asset["name"]
        step = asset["step"]
        wing = asset["wing"]
        lot_size = asset.get("lot_size", 65)

        try:
            df = data_provider.fetch_daily(symbol, days=180)
        except RuntimeError as e:
            logger.warning(f"[MonthlyEngine] No data for {symbol}: {e}. Skipping.")
            send_telegram_message(f"⚠️ <b>Monthly Engine:</b> Skipping {name} — no market data: <code>{e}</code>")
            continue
        if len(df) < 60:
            logger.warning(f"[MonthlyEngine] Insufficient data for {symbol} ({len(df)} candles). Skipping.")
            continue

        x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close']].reset_index(drop=True)
        x_timestamp = df.iloc[-lookback:]['timestamps'].reset_index(drop=True)

        last_date = x_timestamp.iloc[-1]
        y_timestamp = pd.Series([last_date + pd.Timedelta(days=i) for i in range(1, pred_len + 1)])

        try:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=0.8,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
        except Exception as e:
            logger.warning(f"[MonthlyEngine] Model prediction failed for {symbol}: {e}. Skipping asset.")
            send_telegram_message(f"⚠️ <b>Monthly Engine:</b> Skipping {name} — model prediction failed: <code>{e}</code>")
            continue

        curr_close = round(float(x_df["close"].iloc[-1]), 2)
        proj_close = round(float(pred_df["close"].iloc[-1]), 2)
        support = round(float(pred_df["low"].min()), 2)
        resistance = round(float(pred_df["high"].max()), 2)

        # 1. Kronos Directional Return %
        kronos_ret = round(((proj_close - curr_close) / curr_close) * 100.0, 2)

        # 2. Monthly Multi-Asset LSTM Prediction %
        lstm_val = lstm_predictor.predict_return(df, seq_len=min(60, len(df)))
        lstm_ret = round(lstm_val * 100.0, 2)

        # 3. Ensemble Signal Logic
        if kronos_ret >= 1.0 and lstm_ret >= 0.5:
            ensemble_bias = "Bullish"
            confidence = "HIGH"
        elif kronos_ret <= -1.0 and lstm_ret <= -0.5:
            ensemble_bias = "Bearish"
            confidence = "HIGH"
        elif abs(kronos_ret) < 1.0 and abs(lstm_ret) < 0.8:
            ensemble_bias = "Neutral"
            confidence = "HIGH"
        elif (kronos_ret * lstm_ret < 0) and (abs(kronos_ret) >= 0.8 or abs(lstm_ret) >= 0.8):
            ensemble_bias = "Neutral"
            confidence = "DIVERGENT"
        else:
            combined = kronos_ret + lstm_ret
            ensemble_bias = "Bullish" if combined > 0.5 else ("Bearish" if combined < -0.5 else "Neutral")
            confidence = "MODERATE"

        # Calculate range containment probability
        prob = calculate_range_probability(
            current_price=curr_close,
            support=support,
            resistance=resistance,
            hist_prices=df["close"],
            horizon_days=pred_len
        )

        # If models diverge, widen containment wings / require higher safety
        effective_wing = wing if confidence != "DIVERGENT" else int(wing * 1.2)

        # Align target for option spread strategy
        if ensemble_bias == "Bullish":
            target_for_strat = curr_close * (1.0 + max(0.012, (kronos_ret + lstm_ret) / 200.0))
        elif ensemble_bias == "Bearish":
            target_for_strat = curr_close * (1.0 + min(-0.012, (kronos_ret + lstm_ret) / 200.0))
        else:
            target_for_strat = curr_close

        # Determine option selling strategy & exact strikes
        strategy_info = determine_option_strategy(
            current_price=curr_close,
            support=support,
            resistance=resistance,
            proj_target=target_for_strat,
            probability=prob,
            step=step,
            wing_width=effective_wing
        )

        item = {
            "symbol": symbol,
            "name": name,
            "lot_size": lot_size,
            "current_price": curr_close,
            "support": support,
            "resistance": resistance,
            "proj_target": proj_close,
            "kronos_ret": kronos_ret,
            "lstm_ret": lstm_ret,
            "ensemble_bias": ensemble_bias,
            "confidence": confidence,
            "expected_return_pct": round((kronos_ret + lstm_ret) / 2.0, 2),
            "probability": prob,
            "bias": strategy_info.get("bias", ensemble_bias),
            "strategy": strategy_info.get("name", "No Trade"),
            "legs": strategy_info.get("legs", [])
        }
        forecast_results.append(item)

        # Console print
        print(f"[{symbol}] {name} (Lot Size: {lot_size})")
        print(f"  • Spot: ₹{curr_close:,.2f}  |  30D Range: ₹{support:,.2f} to ₹{resistance:,.2f}")
        print(f"  • AI Signals: Kronos {kronos_ret:>+5.2f}% | LSTM {lstm_ret:>+5.2f}% -> Ensemble: {ensemble_bias} ({confidence})")
        print(f"  • In-Range Probability: {prob:.1f}%  |  Strategy: {item['strategy']}")
        if item["legs"]:
            for leg in item["legs"]:
                print(f"      [{leg['action']}] {leg['strike']} {leg['type']} ({leg['desc']})")
        print("-" * 90)

    # Dispatch to Telegram if enabled
    if send_telegram and forecast_results:
        msg = format_options_report(forecast_results, timeframe="Monthly")
        sent = send_telegram_message(msg)
        if sent:
            print("[INFO] Telegram alert dispatched successfully!")
        else:
            print("[INFO] Telegram alert skipped (set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in .env).")

    print("\n[DONE] Monthly option selling desk run completed.\n")
    return forecast_results


if __name__ == "__main__":
    run_monthly_analysis(send_telegram=True)
