import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))
sys.path.append(str(CUR_DIR / "kronos_lib"))

from data_provider import DataProvider
from nifty_direction_engine.hybrid_engine import HybridSignalEngine
from nifty_option_lookup import get_current_tuesday_expiry, get_nifty_option_symbol_and_quote
from telegram_notifier import send_telegram_message, format_intraday_report, format_intraday_exit_report
from event_filter import check_event_risk

logger = logging.getLogger(__name__)

ACTIVE_POSITION_FILE = CUR_DIR / "data_cache" / "active_intraday_position.json"

# Nifty 50 parameters
NIFTY_SYMBOL = "^NSEI"
STEP = 50
OTM_DISTANCE = 200  # 4-OTM strikes buffer = 4 * 50 = 200 points


def save_active_position(position_data: Dict[str, Any]):
    ACTIVE_POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_POSITION_FILE, "w", encoding="utf-8") as f:
        json.dump(position_data, f, indent=2, default=str)


def load_active_position() -> Optional[Dict[str, Any]]:
    if ACTIVE_POSITION_FILE.exists():
        try:
            with open(ACTIVE_POSITION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[DailyEngine] Could not read active position file: {e}")
    return None


def clear_active_position():
    if ACTIVE_POSITION_FILE.exists():
        try:
            ACTIVE_POSITION_FILE.unlink()
        except Exception:
            pass


def run_intraday_entry(send_telegram: bool = True) -> Dict[str, Any]:
    """
    Executes at 10:00 AM IST on all 5 trading days:
    1. Fetches recent 5m Nifty candles up to 10:00 AM.
    2. Runs Hybrid Engine (Kronos + trained LSTM + VWAP acceptance).
    3. Selects 4-OTM strike based on direction:
       - BEARISH: SELL 4-OTM CE (Spot + 200)
       - BULLISH: SELL 4-OTM PE (Spot - 200)
       - SIDEWAYS: SELL BOTH 4-OTM CE and 4-OTM PE (Strangle)
    4. Fetches live entry option premium quotes from Kite for current Tuesday weekly expiry.
    5. Saves active trade to cache and dispatches Telegram alert.
    """
    print("\n" + "=" * 95)
    print("  ⚡  10:00 AM NIFTY 50 HYBRID OPTION SELLING DESK (ENTRY)")
    print("  Schedule: Monday to Friday (All 5 Days) | Rule: 4-OTM (200-Point Buffer)")
    print("  Expiry: Current Week Tuesday | Exit Window: 3:00 PM IST (Full Daily Decay)")
    print("=" * 95 + "\n")

    event_note = check_event_risk()
    if event_note:
        print(f"  ⚠️  HIGH IMPACT EVENT TODAY: {event_note}\n")

    dp = DataProvider()
    engine = HybridSignalEngine(use_kronos=True, use_lstm=True)

    df_5m = dp.fetch_intraday_5m(NIFTY_SYMBOL, days=5, use_cache=False)
    if df_5m is None or len(df_5m) < 40:
        logger.error("[DailyEngine] Insufficient 5m data. Skipping entry.")
        return {}

    # Slice lookback of 100 bars up to 10:00 AM
    lookback = min(100, len(df_5m))
    slice_df = df_5m.iloc[-lookback:].copy().reset_index(drop=True)

    spot_10am = float(slice_df["close"].iloc[-1])
    atm_strike = int(round(spot_10am / STEP) * STEP)

    # Evaluate Hybrid Signal at 10:00 AM
    signal_res = engine.generate_signal(slice_df, deadband=0.0015)
    bias = signal_res.get("final_bias", "SIDEWAYS_NO_TRADE")

    target_tuesday = get_current_tuesday_expiry(dp.kite)
    expiry_str = target_tuesday.strftime("%d-%b-%Y") if target_tuesday else "Nearest Tuesday"

    legs = []

    if bias == "BEARISH":
        sold_strike = atm_strike + OTM_DISTANCE
        tsym, ltp = get_nifty_option_symbol_and_quote(dp.kite, sold_strike, "CE", target_tuesday)
        legs.append({
            "action": "SELL",
            "strike": sold_strike,
            "type": "CE",
            "tradingsymbol": tsym,
            "entry_premium": ltp,
            "desc": "4-OTM Resistance Ceiling (+200 pts)"
        })
        action_desc = f"SELL {sold_strike} CE"

    elif bias == "BULLISH":
        sold_strike = atm_strike - OTM_DISTANCE
        tsym, ltp = get_nifty_option_symbol_and_quote(dp.kite, sold_strike, "PE", target_tuesday)
        legs.append({
            "action": "SELL",
            "strike": sold_strike,
            "type": "PE",
            "tradingsymbol": tsym,
            "entry_premium": ltp,
            "desc": "4-OTM Support Floor (-200 pts)"
        })
        action_desc = f"SELL {sold_strike} PE"

    else:  # SIDEWAYS / NO_TRADE -> Sell 4-OTM Strangle (both CE and PE)
        bias = "SIDEWAYS"
        ce_strike = atm_strike + OTM_DISTANCE
        pe_strike = atm_strike - OTM_DISTANCE

        ce_sym, ce_ltp = get_nifty_option_symbol_and_quote(dp.kite, ce_strike, "CE", target_tuesday)
        pe_sym, pe_ltp = get_nifty_option_symbol_and_quote(dp.kite, pe_strike, "PE", target_tuesday)

        legs.append({
            "action": "SELL",
            "strike": ce_strike,
            "type": "CE",
            "tradingsymbol": ce_sym,
            "entry_premium": ce_ltp,
            "desc": "4-OTM Call Leg (+200 pts)"
        })
        legs.append({
            "action": "SELL",
            "strike": pe_strike,
            "type": "PE",
            "tradingsymbol": pe_sym,
            "entry_premium": pe_ltp,
            "desc": "4-OTM Put Leg (-200 pts)"
        })
        action_desc = f"SELL {pe_strike} PE & {ce_strike} CE (Strangle)"

    position_payload = {
        "date": str(datetime.now().date()),
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spot_10am": spot_10am,
        "atm_strike": atm_strike,
        "bias": bias,
        "expiry_str": expiry_str,
        "target_tuesday": str(target_tuesday) if target_tuesday else "",
        "legs": legs
    }
    save_active_position(position_payload)

    report_item = {
        "name": "NIFTY 50 Index",
        "symbol": "^NSEI",
        "current_price": spot_10am,
        "intraday_bias": bias,
        "intraday_action": action_desc,
        "expiry_str": expiry_str,
        "legs": legs,
        "event_note": event_note or ""
    }

    if send_telegram:
        msg = format_intraday_report([report_item])
        send_telegram_message(msg)

    print(f"  • 10:00 AM Spot: ₹{spot_10am:,.2f}")
    print(f"  • Bias: {bias} | Strategy: {action_desc}")
    for leg in legs:
        print(f"    -> {leg['action']} {leg['tradingsymbol'] or leg['strike']} @ LTP ₹{leg['entry_premium']:.2f}")

    print("\n[DONE] 10:00 AM Entry Alert processed.\n")
    return position_payload


def run_intraday_exit(send_telegram: bool = True) -> Dict[str, Any]:
    """
    Executes at 3:00 PM IST:
    1. Loads active trade entered at 10:00 AM.
    2. Fetches live 3:00 PM spot and option quotes.
    3. Calculates exact Option Premium PnL:
       For Sold Option: Points Gained = (Entry Premium - Exit Premium)
    4. Dispatches exit alert via Telegram with full breakdown.
    5. Clears active position.
    """
    print("\n" + "=" * 95)
    print("  🏁  3:00 PM NIFTY 50 OPTION DESK (EXIT & PnL)")
    print("=" * 95 + "\n")

    pos = load_active_position()
    if not pos:
        logger.warning("[DailyEngine] No active intraday position found to exit.")
        return {}

    dp = DataProvider()
    df_5m = dp.fetch_intraday_5m(NIFTY_SYMBOL, days=1, use_cache=False)
    spot_1500 = float(df_5m["close"].iloc[-1]) if df_5m is not None and not df_5m.empty else pos.get("spot_10am", 0.0)

    legs_pnl = []
    total_pts = 0.0

    for leg in pos.get("legs", []):
        tradingsymbol = leg.get("tradingsymbol", "")
        entry_prem = float(leg.get("entry_premium", 0.0))
        exit_prem = entry_prem # Fallback

        if dp.kite and tradingsymbol:
            try:
                key = f"NFO:{tradingsymbol}"
                quote = dp.kite.quote(key)
                exit_prem = float(quote[key]["last_price"])
            except Exception as e:
                logger.warning(f"[DailyEngine] Could not fetch exit quote for {tradingsymbol}: {e}")

        # When selling options: Profit = Entry - Exit
        points_gained = entry_prem - exit_prem
        total_pts += points_gained

        legs_pnl.append({
            "action": leg.get("action", "SELL"),
            "strike": leg.get("strike", 0),
            "type": leg.get("type", "CE"),
            "tradingsymbol": tradingsymbol,
            "entry_premium": entry_prem,
            "exit_premium": exit_prem,
            "points_gained": points_gained
        })

    exit_payload = {
        "date": pos.get("date", ""),
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spot_10am": pos.get("spot_10am", 0.0),
        "spot_1500": spot_1500,
        "bias": pos.get("bias", "SIDEWAYS"),
        "legs_pnl": legs_pnl,
        "total_points_captured": total_pts
    }

    if send_telegram:
        msg = format_intraday_exit_report(exit_payload)
        send_telegram_message(msg)

    print(f"  • Entry Spot: ₹{pos.get('spot_10am', 0.0):,.2f} -> Exit Spot: ₹{spot_1500:,.2f}")
    for leg in legs_pnl:
        print(f"    -> {leg['tradingsymbol']}: Entry ₹{leg['entry_premium']:.2f} | Exit ₹{leg['exit_premium']:.2f} | PnL: {leg['points_gained']:+.2f} pts")
    print(f"  • Net Points Captured: {total_pts:+.2f} pts (₹{total_pts * 25:+,.2f} per lot)")

    clear_active_position()
    print("\n[DONE] 3:00 PM Exit processed successfully.\n")
    return exit_payload


# Backward-compatible wrapper for external scripts
def run_intraday_analysis(send_telegram: bool = True, target_date: Optional[Any] = None) -> List[Dict[str, Any]]:
    return [run_intraday_entry(send_telegram=send_telegram)]


if __name__ == "__main__":
    run_intraday_entry(send_telegram=False)
