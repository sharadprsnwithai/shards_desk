from datetime import date
from scheduler import is_last_day_of_month

def test_is_last_day_of_month_various_months():
    # January 31 -> True
    assert is_last_day_of_month(date(2025, 1, 31)) is True
    assert is_last_day_of_month(date(2025, 1, 30)) is False

    # February in non-leap year (2025) -> Feb 28 is True
    assert is_last_day_of_month(date(2025, 2, 28)) is True
    assert is_last_day_of_month(date(2025, 2, 27)) is False

    # February in leap year (2024, 2028)
    assert is_last_day_of_month(date(2024, 2, 29)) is True
    assert is_last_day_of_month(date(2024, 2, 28)) is False

    # April (30 days) -> April 30 is True
    assert is_last_day_of_month(date(2025, 4, 30)) is True
    assert is_last_day_of_month(date(2025, 4, 29)) is False

    # December 31 -> True
    assert is_last_day_of_month(date(2025, 12, 31)) is True
    assert is_last_day_of_month(date(2025, 12, 30)) is False

def test_docker_entrypoint_has_monthly_cron():
    import os
    entrypoint_path = os.path.join(os.path.dirname(__file__), "..", "docker-entrypoint.sh")
    assert os.path.exists(entrypoint_path)
    with open(entrypoint_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "scheduler.py" in content
    assert "monthly" in content


def test_scheduler_triggers_monthly_retrain_and_run():
    from unittest.mock import MagicMock
    import pytz
    from datetime import datetime
    from scheduler import check_scheduled_triggers

    tz = pytz.timezone("Asia/Kolkata")
    state = {
        "last_entry_date": None,
        "last_exit_date": None,
        "last_monthly_train_date": None,
        "last_monthly_run_date": None,
    }

    mock_entry = MagicMock()
    mock_exit = MagicMock()
    mock_train = MagicMock()
    mock_monthly = MagicMock()

    # 1. Test 4:00 PM IST on last day of month -> triggers retrain
    t_1600 = tz.localize(datetime(2025, 1, 31, 16, 0, 0))
    check_scheduled_triggers(
        t_1600,
        state,
        run_intraday_entry_fn=mock_entry,
        run_intraday_exit_fn=mock_exit,
        train_monthly_fn=mock_train,
        run_monthly_fn=mock_monthly,
    )
    assert mock_train.call_count == 1
    assert state["last_monthly_train_date"] == datetime(2025, 1, 31).date()

    # 2. Test 5:00 PM IST on last day of month -> triggers monthly run
    t_1700 = tz.localize(datetime(2025, 1, 31, 17, 0, 0))
    check_scheduled_triggers(
        t_1700,
        state,
        run_intraday_entry_fn=mock_entry,
        run_intraday_exit_fn=mock_exit,
        train_monthly_fn=mock_train,
        run_monthly_fn=mock_monthly,
    )
    assert mock_monthly.call_count == 1
    assert state["last_monthly_run_date"] == datetime(2025, 1, 31).date()

