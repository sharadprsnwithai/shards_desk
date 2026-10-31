import pytest
from datetime import date
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import daily_engine

def _static_quote_fn(strike, opt_type):
    return (f"NIFTY{strike}{opt_type}", 25.0)

def test_build_legs_sideways_has_hedge_wings():
    legs = daily_engine.build_legs("SIDEWAYS", 24000, _static_quote_fn)
    actions = [l["action"] for l in legs]
    assert actions.count("SELL") == 2 and actions.count("BUY") == 2
    sold = {l["strike"] for l in legs if l["action"] == "SELL"}
    wings = {l["strike"] for l in legs if l["action"] == "BUY"}
    assert sold == {24200, 23800}      # +-200 pts from ATM
    assert wings == {24350, 23650}     # sold CE +150 / sold PE -150
    for w in [l for l in legs if l["action"] == "BUY"]:
        assert "sl_premium" not in w   # wings are exited only by EOD, per plan

def test_build_legs_bullish_sells_put_with_wing():
    legs = daily_engine.build_legs("BULLISH", 24000, _static_quote_fn)
    assert len(legs) == 2
    assert legs[0]["action"] == "SELL" and legs[0]["type"] == "PE" and legs[0]["strike"] == 23800
    assert legs[1]["action"] == "BUY" and legs[1]["type"] == "PE" and legs[1]["strike"] == 23650

def test_build_legs_bearish_sells_call_with_wing():
    legs = daily_engine.build_legs("BEARISH", 24000, _static_quote_fn)
    assert len(legs) == 2
    assert legs[0]["action"] == "SELL" and legs[0]["type"] == "CE" and legs[0]["strike"] == 24200
    assert legs[1]["action"] == "BUY" and legs[1]["type"] == "CE" and legs[1]["strike"] == 24350

def test_monitor_skips_buy_legs(tmp_path, monkeypatch):
    daily_engine.ACTIVE_POSITION_FILE = tmp_path / "pos.json"
    pos = {"date": "2026-10-31", "legs": [
        {"action": "BUY", "strike": 24350, "type": "CE",
         "tradingsymbol": "NIFTY24350CE",
         "entry_premium": 1.0, "sl_premium": 2.0, "target_premium": 0.2,
         "status": "ACTIVE"}]}
    daily_engine.save_active_position(pos)

    mock_kite = MagicMock()
    mock_kite.quote.return_value = {"NFO:NIFTY24350CE": {"last_price": 0.1}}
    mock_dp = MagicMock()
    mock_dp.kite = mock_kite
    monkeypatch.setattr(daily_engine, "DataProvider", lambda: mock_dp)

    triggered = daily_engine.monitor_intraday_sl_tp(send_telegram=False)
    assert triggered == []
    reloaded = daily_engine.load_active_position()
    assert reloaded["legs"][0]["status"] == "ACTIVE"   # BUY leg must never be SL/TP-monitored


def test_event_day_places_no_trade(tmp_path, monkeypatch):
    daily_engine.ACTIVE_POSITION_FILE = tmp_path / "pos.json"
    monkeypatch.setattr(daily_engine, "check_event_risk", lambda: "RBI Policy")
    monkeypatch.setattr(daily_engine.DataProvider, "fetch_intraday_5m",
                        MagicMock(side_effect=AssertionError("must not fetch data on event day")))
    assert daily_engine.run_intraday_entry(send_telegram=False) == {}
    assert not daily_engine.ACTIVE_POSITION_FILE.exists()


def test_daily_intraday_nifty_setup():

    assert daily_engine.NIFTY_SYMBOL == "^NSEI"
    assert daily_engine.STEP == 50
    assert daily_engine.OTM_DISTANCE == 200

def test_daily_intraday_entry_and_exit_flow(tmp_path):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=120, freq='5min')
    dummy_df = pd.DataFrame({
        'timestamps': dates,
        'open': np.linspace(24100, 24200, 120),
        'high': np.linspace(24150, 24250, 120),
        'low': np.linspace(24050, 24150, 120),
        'close': np.linspace(24120, 24220, 120),
        'volume': [50000] * 120
    })

    # Point active position file to temp
    daily_engine.ACTIVE_POSITION_FILE = tmp_path / "test_pos.json"

    with patch.object(daily_engine.DataProvider, "fetch_intraday_5m", return_value=dummy_df):
        with patch.object(daily_engine, "get_nifty_option_symbol_and_quote", return_value=("NIFTY26OCT24400CE", 25.50)):
            # 1. Test Entry at 10:00 AM
            entry_res = daily_engine.run_intraday_entry(send_telegram=False)
            assert "spot_10am" in entry_res
            assert "bias" in entry_res
            assert len(entry_res["legs"]) >= 1

            # 2. Test Exit at 3:00 PM
            exit_res = daily_engine.run_intraday_exit(send_telegram=False)
            assert "spot_1500" in exit_res
            assert "legs_pnl" in exit_res
            assert "total_points_captured" in exit_res
