"""
NSE Trading Calendar
====================
Provides holiday-aware trading day checks using pandas_market_calendars.
"""
from datetime import date
from functools import lru_cache
from typing import List

import pandas_market_calendars as mcal


@lru_cache(maxsize=8)
def get_nse_holidays(year: int) -> List[date]:
    """
    Returns a sorted list of NSE exchange holiday dates for the given year.
    Results are cached per year (max 8 years in memory).
    """
    cal = mcal.get_calendar("NSE")
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    holidays = cal.holidays().holidays  # pandas DatetimeIndex
    import pandas as pd
    year_holidays = [
        pd.Timestamp(d).date() for d in holidays
        if pd.Timestamp(d).year == year
    ]
    return sorted(year_holidays)


def is_nse_trading_day(d: date) -> bool:
    """
    Returns True only when *d* is a weekday (Mon–Fri) and not an NSE
    exchange holiday.  Returns False on weekends and all NSE holidays.
    """
    if d.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    return d not in get_nse_holidays(d.year)
