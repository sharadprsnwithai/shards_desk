from datetime import date
from market_calendar import is_nse_trading_day, get_nse_holidays


def test_weekend_is_not_trading_day():
    assert is_nse_trading_day(date(2025, 1, 4)) is False   # Saturday
    assert is_nse_trading_day(date(2025, 1, 5)) is False   # Sunday


def test_known_nse_holiday_is_not_trading_day():
    # Independence Day 2025 — confirmed in pandas_market_calendars NSE list
    assert is_nse_trading_day(date(2025, 8, 15)) is False


def test_normal_weekday_is_trading_day():
    assert is_nse_trading_day(date(2025, 8, 18)) is True   # Monday after Independence Day


def test_get_nse_holidays_returns_list_of_dates():
    holidays = get_nse_holidays(2025)
    assert isinstance(holidays, list)
    assert all(isinstance(h, date) for h in holidays)
    assert date(2025, 8, 15) in holidays   # Independence Day
