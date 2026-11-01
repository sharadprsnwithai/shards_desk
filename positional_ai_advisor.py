"""
positional_ai_advisor.py
Daily 3:10 PM OpenRouter AI Quantitative Advisor for Zerodha Kite Positional Trades.

Queries active net positions from Zerodha Kite, loads monthly desk state, 
combines with live market spot prices, and sends an expert prompt to OpenRouter 
(Claude 3.5 Sonnet / GPT-4o) to evaluate risk, suggest adjustments, and maximize yield.
"""
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import html
from dotenv import load_dotenv

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from data_provider import DataProvider
from telegram_notifier import send_telegram_message

logger = logging.getLogger(__name__)

ENV_PATH = CUR_DIR / ".env"


def _load_env():
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)


def run_positional_ai_advisor(send_telegram: bool = True) -> Dict[str, Any]:
    """
    Executes at 3:10 PM IST on every working day:
    1. Connects to Zerodha Kite to fetch live net portfolio positions.
    2. Fetches latest spot prices for Nifty, Bank Nifty, Sensex, Reliance, TCS, HDFC Bank.
    3. Calls OpenRouter API with professional quantitative options risk management prompt.
    4. Dispatches AI recommendations to Telegram.
    """
    _load_env()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("AI_MODEL", "deepseek/deepseek-chat").strip()

    print("\n" + "=" * 90)
    print("  AI  OPENROUTER AI QUANTITATIVE POSITIONAL ADVISOR (3:10 PM DAILY EVAL)")
    print("=" * 90 + "\n")

    if not api_key:
        logger.warning("[AI-Advisor] OPENROUTER_API_KEY missing in .env. Skipping AI advisory.")
        if send_telegram:
            send_telegram_message("⚠️ <b>AI Advisor Skipped:</b> <code>OPENROUTER_API_KEY</code> not configured in .env.")
        return {"status": "API_KEY_MISSING"}

    # 1. Fetch Zerodha Kite Net Positions (Single Source of Truth)
    dp = DataProvider()
    kite = dp.kite
    kite_positions = []
    if kite:
        try:
            pos_response = kite.positions()
            kite_positions = pos_response.get("net", [])
        except Exception as e:
            logger.warning(f"[AI-Advisor] Could not fetch Kite positions: {e}")

    # 3. Fetch current spot prices for core assets
    spot_prices = {}
    symbols = ["^NSEI", "^NSEBANK", "^BSESN", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    for sym in symbols:
        try:
            df = dp.fetch_daily(sym, days=5)
            if not df.empty:
                spot_prices[sym] = float(df["close"].iloc[-1])
        except Exception:
            pass

    # 4. Construct professional quantitative prompt
    portfolio_context = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "live_kite_net_positions": [
            {
                "tradingsymbol": p.get("tradingsymbol"),
                "exchange": p.get("exchange"),
                "product": p.get("product"),
                "quantity": p.get("quantity"),
                "average_price": p.get("average_price"),
                "last_price": p.get("last_price"),
                "pnl": p.get("pnl"),
                "m2m": p.get("m2m")
            }
            for p in kite_positions if p.get("quantity", 0) != 0
        ],
        "current_spot_prices": spot_prices
    }

    prompt = (
        "You are an elite quantitative options risk manager and portfolio strategist managing a "
        "positional option selling desk on Indian markets (Zerodha Kite). "
        "Analyze the following live portfolio state, current spot prices, and active credit spreads/wings:\n\n"
        f"{json.dumps(portfolio_context, indent=2)}\n\n"
        "Provide a concise, professional risk assessment report covering:\n"
        "1. **Delta Exposure & Strike Drift:** Are any short strikes currently threatened or within 1.5% of spot?\n"
        "2. **Tail Risk & Hedging:** Are hedge wings functioning adequately to cap black swan risk?\n"
        "3. **Actionable Adjustments:** Specific, step-by-step recommendations to reduce risk, roll untested sides, or convert spreads (e.g. to Iron Butterflies) to maximize yield.\n"
        "Keep the output structured, institutional, and direct."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/sharad/option-selling-desk",
        "X-Title": "Sharads Option Selling Desk"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert quantitative derivatives risk manager."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1200
    }

    ai_response_text = ""
    try:
        logger.info(f"[AI-Advisor] Sending portfolio payload to OpenRouter ({model})...")
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        resp_json = resp.json()
        if "choices" in resp_json:
            ai_response_text = resp_json["choices"][0]["message"]["content"]
        else:
            ai_response_text = f"OpenRouter Error Response: {json.dumps(resp_json)}"
    except Exception as e:
        logger.error(f"[AI-Advisor] OpenRouter API call failed: {e}")
        ai_response_text = f"API Request Failed: {e}"

    # 5. Format and Dispatch Telegram Message
    telegram_msg = (
        f"🤖 <b>OPENROUTER AI ADVISOR (3:10 PM EVAL)</b>\n"
        f"Model: <code>{model}</code>\n\n"
        f"{html.escape(ai_response_text).replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>').replace('&lt;code&gt;', '<code>').replace('&lt;/code&gt;', '</code>')}"
    )

    if send_telegram:
        send_telegram_message(telegram_msg)

    print(ai_response_text)
    print("\n[DONE] Positional AI Advisor run completed.\n")
    return {"status": "SUCCESS", "ai_advice": ai_response_text}


if __name__ == "__main__":
    run_positional_ai_advisor(send_telegram=True)
