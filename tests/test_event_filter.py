from datetime import date
from unittest.mock import patch
from event_filter import should_skip_trading


def test_skip_on_nse_holiday():
    with patch("event_filter.is_nse_trading_day", return_value=False):
        assert should_skip_trading(date(2025, 8, 15)) is True


def test_skip_on_high_impact_event():
    with patch("event_filter.is_nse_trading_day", return_value=True):
        # RBI MPC day in the hardcoded calendar
        assert should_skip_trading(date(2025, 2, 7)) is True


def test_normal_day_not_skipped():
    with patch("event_filter.is_nse_trading_day", return_value=True):
        assert should_skip_trading(date(2025, 8, 18)) is False
