import os
import pytest
from telegram_notifier import format_options_report, send_telegram_message
from config import Config

def test_config_loads_defaults(monkeypatch):
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    cfg = Config.load()
    assert cfg.kite_api_key == ""
    assert cfg.telegram_bot_token == ""

def test_format_options_report_monthly():
    sample_forecast = [{
        "name": "NIFTY 50 Index",
        "symbol": "^NSEI",
        "current_price": 24180.0,
        "support": 23750.0,
        "resistance": 24650.0,
        "proj_target": 24320.0,
        "expected_return_pct": 0.58,
        "probability": 78.5,
        "bias": "Neutral",
        "strategy": "Iron Condor",
        "legs": [
            {"action": "SELL", "strike": 23750, "type": "PE", "desc": "Support Leg"},
            {"action": "BUY",  "strike": 23550, "type": "PE", "desc": "Put Hedge Wing"},
            {"action": "SELL", "strike": 24650, "type": "CE", "desc": "Resistance Leg"},
            {"action": "BUY",  "strike": 24850, "type": "CE", "desc": "Call Hedge Wing"}
        ]
    }]
    msg = format_options_report(sample_forecast, timeframe="Monthly")
    assert "NIFTY 50" in msg
    assert "Iron Condor" in msg
    assert "23750" in msg
    assert "78.5%" in msg
    assert "SELL 23750 PE" in msg

def test_send_telegram_message_safely_skips_without_creds(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    # Must return False and never crash
    res = send_telegram_message("Test Alert")
    assert res is False
