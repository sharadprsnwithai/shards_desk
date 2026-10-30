import pytest
from datetime import date
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import daily_engine

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
