import pytest
import pandas as pd
from data_provider import DataProvider

def test_symbol_mapping():
    provider = DataProvider(kite_api_key=None, kite_access_token=None)
    assert provider.to_kite_symbol("^NSEI") == "NIFTY 50"
    assert provider.to_kite_symbol("RELIANCE.NS") == "RELIANCE"
    assert provider.to_kite_symbol("TCS.NS") == "TCS"
    assert provider.to_kite_symbol("HDFCBANK.NS") == "HDFCBANK"

def test_banknifty_instrument_mapping():
    dp = DataProvider(kite_api_key=None, kite_access_token=None)
    assert dp.get_instrument_token("^NSEBANK") == 260105
    assert dp.to_kite_symbol("^NSEBANK") == "NIFTY BANK"

def test_fetch_daily_fallback():
    provider = DataProvider(kite_api_key=None, kite_access_token=None)
    df = provider.fetch_daily("^NSEI", days=90)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"timestamps", "open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 30
    # Ensure timestamps are datetime
    assert pd.api.types.is_datetime64_any_dtype(df["timestamps"])

def test_fetch_intraday_fallback():
    provider = DataProvider(kite_api_key=None, kite_access_token=None)
    df = provider.fetch_intraday_5m("^NSEI", days=5)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"timestamps", "open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df.columns))

def test_relogin_retry_5_times_then_fallback(monkeypatch):
    from unittest.mock import MagicMock
    mock_kite = MagicMock()
    # Mock TokenException simulation
    mock_kite.historical_data.side_effect = Exception("TokenException: Session invalid or expired")

    provider = DataProvider(kite_api_key="mock_key", kite_access_token="mock_token")
    provider.kite = mock_kite

    attempt_counter = [0]
    def mock_relogin():
        attempt_counter[0] += 1
        return False  # Relogin fails

    monkeypatch.setattr(provider, "attempt_relogin", mock_relogin)

    # Calling fetch_daily with use_cache=False should attempt relogin exactly 5 times, then gracefully fall back to yfinance
    df = provider.fetch_daily("^NSEI", days=30, use_cache=False)
    assert attempt_counter[0] == 5
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_relogin_succeeds_on_retry(monkeypatch):
    from unittest.mock import MagicMock
    mock_kite = MagicMock()
    
    # Fails twice with expired session, succeeds on 3rd attempt
    call_count = [0]
    def mock_hist(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("403 Forbidden: Session expired")
        return [
            {"date": "2026-10-25 09:15:00", "open": 24000, "high": 24100, "low": 23950, "close": 24050, "volume": 1000}
        ]

    mock_kite.historical_data.side_effect = mock_hist
    provider = DataProvider(kite_api_key="mock_key", kite_access_token="mock_token")
    provider.kite = mock_kite

    relogin_count = [0]
    def mock_relogin():
        relogin_count[0] += 1
        return True

    monkeypatch.setattr(provider, "attempt_relogin", mock_relogin)

    df = provider.fetch_daily("^NSEI", days=30, use_cache=False)
    assert relogin_count[0] == 2  # Relogged in twice before succeeding
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == 24050



def test_fetch_daily_raises_when_both_sources_fail_and_synthetic_disabled():
    from unittest.mock import patch
    dp = DataProvider(kite_api_key=None, kite_access_token=None)
    dp.kite = None
    with patch("data_provider.yf.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="No market data available"):
            dp.fetch_daily("^NSEI", days=30, use_cache=False, allow_synthetic=False)


def test_fetch_daily_returns_synthetic_when_explicitly_allowed():
    from unittest.mock import patch
    dp = DataProvider(kite_api_key=None, kite_access_token=None)
    dp.kite = None
    with patch("data_provider.yf.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = pd.DataFrame()
        result = dp.fetch_daily("^NSEI", days=30, use_cache=False, allow_synthetic=True)
    assert len(result) > 0


def test_fetch_intraday_5m_raises_when_both_fail():
    from unittest.mock import patch
    dp = DataProvider(kite_api_key=None, kite_access_token=None)
    dp.kite = None
    with patch("data_provider.yf.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = pd.DataFrame()
        with pytest.raises(RuntimeError, match="No market data available"):
            dp.fetch_intraday_5m("^NSEI", days=3, use_cache=False, allow_synthetic=False)
