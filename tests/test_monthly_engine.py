import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

def test_monthly_engine_execution_flow(monkeypatch):
    import monthly_engine
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=120, freq='D')
    dummy_df = pd.DataFrame({
        'timestamps': dates,
        'open': np.linspace(24000, 24200, 120),
        'high': np.linspace(24100, 24300, 120),
        'low': np.linspace(23900, 24100, 120),
        'close': np.linspace(24000, 24250, 120),
        'volume': [100000] * 120
    })
    
    with patch.object(monthly_engine.DataProvider, "fetch_daily", return_value=dummy_df):
        with patch.object(monthly_engine, "get_kronos_predictor") as mock_get_pred:
            mock_predictor = MagicMock()
            mock_get_pred.return_value = mock_predictor
            
            # Predict returns 30 days
            mock_predictor.predict.return_value = pd.DataFrame({
                'open': [24250.0]*30,
                'high': [24600.0]*30,
                'low': [23800.0]*30,
                'close': [24400.0]*30,
            })
            
            results = monthly_engine.run_monthly_analysis(send_telegram=False)
            assert len(results) == 6
            symbols = [r["symbol"] for r in results]
            assert "^NSEI" in symbols
            assert "^NSEBANK" in symbols
            assert "^BSESN" in symbols
            assert "RELIANCE.NS" in symbols
            assert "TCS.NS" in symbols
            assert "HDFCBANK.NS" in symbols

            # Verify fields on NIFTY
            nifty_res = next(r for r in results if r["symbol"] == "^NSEI")
            assert nifty_res["lot_size"] == 65
            assert "current_price" in nifty_res
            assert "support" in nifty_res
            assert "resistance" in nifty_res
            assert "probability" in nifty_res
            assert "strategy" in nifty_res
            assert "legs" in nifty_res
            assert "lstm_ret" in nifty_res
            assert "kronos_ret" in nifty_res
            assert "ensemble_bias" in nifty_res
            assert "confidence" in nifty_res
