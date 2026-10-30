import numpy as np
import pandas as pd
import pytest
from monthly_direction_engine.feature_extractor import extract_positional_features

def test_extract_positional_features():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = np.linspace(20000, 22000, 100) + np.random.normal(0, 50, 100)
    df = pd.DataFrame({
        "timestamps": dates,
        "open": prices * 0.998,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": [100000] * 100
    })
    feats = extract_positional_features(df)
    assert len(feats) == 100
    expected_cols = ["norm_ret", "hl_range", "norm_atr", "norm_rsi", "sma20_dev", "sma50_dev", "realized_vol"]
    for col in expected_cols:
        assert col in feats.columns
    # Check no NaNs after backfilling initial window
    assert not feats[expected_cols].isna().any().any()


def test_monthly_lstm_model_forward():
    import torch
    from monthly_direction_engine.monthly_lstm_model import MonthlyLSTMModel
    model = MonthlyLSTMModel(input_dim=7, hidden_dim=64, num_layers=2)
    dummy_input = torch.randn(4, 60, 7)
    out = model(dummy_input)
    assert out.shape == (4, 1)


def test_monthly_lstm_predictor_inference():
    import torch
    from monthly_direction_engine.monthly_lstm_model import MonthlyLSTMPredictor
    predictor = MonthlyLSTMPredictor(weights_path=None)  # Tests uninitialized fallback
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    prices = np.linspace(24000, 24500, 80)
    df = pd.DataFrame({
        "timestamps": dates,
        "open": prices, "high": prices * 1.01, "low": prices * 0.99, "close": prices, "volume": [1000] * 80
    })
    pred_ret = predictor.predict_return(df, seq_len=60)
    assert isinstance(pred_ret, float)


def test_load_trained_weights():
    from pathlib import Path
    from monthly_direction_engine.monthly_lstm_model import MonthlyLSTMPredictor
    weights_path = Path("monthly_direction_engine/monthly_lstm_weights.pt")
    assert weights_path.exists()
    predictor = MonthlyLSTMPredictor(weights_path=str(weights_path))
    assert predictor.is_loaded is True
    assert predictor.mean is not None


