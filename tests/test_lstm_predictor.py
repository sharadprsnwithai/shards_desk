import pytest
from nifty_direction_engine.lstm_predictor import LSTMDirectionPredictor


def test_lstm_raises_when_weights_missing(tmp_path):
    missing = str(tmp_path / "nonexistent_weights.pt")
    with pytest.raises(RuntimeError, match="LSTM weights file not found"):
        LSTMDirectionPredictor(weights_path=missing)


def test_lstm_loads_successfully_when_weights_present():
    import os
    weights = "nifty_direction_engine/lstm_nifty_weights.pt"
    if not os.path.exists(weights):
        pytest.skip("Weights file not present in this env")
    predictor = LSTMDirectionPredictor(weights_path=weights)
    assert predictor is not None
