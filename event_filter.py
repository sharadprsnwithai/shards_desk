from datetime import datetime, date
from typing import Optional, Dict

# Known high-impact scheduled dates (RBI Monetary Policy Committee, Union Budget, etc.)
# Can be updated or expanded with NSE corporate earnings dates
MAJOR_EVENT_CALENDAR: Dict[str, str] = {
    "2025-02-01": "Union Budget Presentation",
    "2025-02-07": "RBI MPC Monetary Policy Decision",
    "2025-04-09": "RBI MPC Monetary Policy Decision",
    "2025-06-06": "RBI MPC Monetary Policy Decision",
    "2025-08-08": "RBI MPC Monetary Policy Decision",
    "2025-10-08": "RBI MPC Monetary Policy Decision",
    "2025-12-05": "RBI MPC Monetary Policy Decision",
    "2026-02-01": "Union Budget Presentation",
    "2026-02-06": "RBI MPC Monetary Policy Decision",
    "2026-04-08": "RBI MPC Monetary Policy Decision",
    "2026-06-05": "RBI MPC Monetary Policy Decision",
    "2026-08-07": "RBI MPC Monetary Policy Decision",
    "2026-10-09": "RBI MPC Monetary Policy Decision",
    "2026-12-04": "RBI MPC Monetary Policy Decision",
}


def check_event_risk(target_date: Optional[date] = None) -> Optional[str]:
    """
    Checks if the target date coincides with a major macroeconomic event.
    Returns the event description or None if it's a standard trading session.
    """
    check_d = target_date or datetime.now().date()
    date_str = check_d.strftime("%Y-%m-%d")
    return MAJOR_EVENT_CALENDAR.get(date_str, None)
