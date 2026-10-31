import pandas as pd
import numpy as np
from unittest.mock import patch
from nifty_direction_engine.hybrid_engine import HybridSignalEngine


def _make_df(n=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    base = 24000.0
    closes = base + np.cumsum(np.random.normal(0, 5, n))
    return pd.DataFrame({
        "timestamps": idx, "open": closes * 0.999,
        "high": closes * 1.001, "low": closes * 0.999, "close": closes,
        "volume": np.ones(n) * 1000
    })


def test_generate_signal_returns_no_trade_when_lstm_weights_missing(tmp_path):
    missing = str(tmp_path / "no_weights.pt")
    engine = HybridSignalEngine(use_kronos=False, use_lstm=True, weights_path=missing)
    result = engine.generate_signal(_make_df())
    assert result["final_bias"] == "SIDEWAYS_NO_TRADE"
    reason = result.get("reason", "").lower()
    assert "weights" in reason or "unavailable" in reason
