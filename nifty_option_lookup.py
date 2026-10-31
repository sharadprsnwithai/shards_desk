import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, Tuple
from data_provider import DataProvider
from market_calendar import is_nse_trading_day

logger = logging.getLogger(__name__)

def get_current_tuesday_expiry(kite_client) -> Optional[date]:
    """
    Finds the closest upcoming Tuesday expiry for NIFTY options.
    If today is Tuesday, it returns today's expiry.
    If Tuesday is an exchange holiday, checks for Monday pre-holiday expiry.
    """
    today = datetime.now().date()
    try:
        if kite_client:
            instruments = kite_client.instruments("NFO")
            # Include Tuesday (weekday 1) and Monday (weekday 0 if Tuesday holiday)
            tuesday_expiries = sorted(list(set(
                i["expiry"] for i in instruments 
                if i["name"] == "NIFTY" and i["instrument_type"] in ["CE", "PE"] and i["expiry"].weekday() in (0, 1)
            )))
            
            valid_expiries = [e for e in tuesday_expiries if e >= today]
            if valid_expiries:
                return valid_expiries[0]
            elif tuesday_expiries:
                return tuesday_expiries[-1]
    except Exception as e:
        logger.warning(f"[OptionLookup] Failed to find Tuesday expiry: {e}")
        
    # Fallback to computing closest Tuesday date, shifted for NSE holidays
    days_ahead = (1 - today.weekday()) % 7 # 1 = Tuesday
    target_d = today + timedelta(days=days_ahead)
    try:
        while not is_nse_trading_day(target_d) and target_d > today:
            target_d -= timedelta(days=1)
    except Exception:
        pass
    return target_d


def get_nifty_option_symbol_and_quote(
    kite_client,
    strike: int,
    opt_type: str,
    target_expiry: Optional[date] = None
) -> Tuple[Optional[str], float]:
    """
    Returns (tradingsymbol, last_price) for given strike, type (CE/PE), and Tuesday expiry.
    """
    if kite_client is None:
        return None, 0.0

    try:
        if target_expiry is None:
            target_expiry = get_current_tuesday_expiry(kite_client)

        instruments = kite_client.instruments("NFO")
        matching = [
            i for i in instruments
            if i["name"] == "NIFTY" 
            and i["instrument_type"] == opt_type
            and abs(float(i["strike"]) - float(strike)) < 1.0
            and i["expiry"] == target_expiry
        ]

        if not matching:
            # Fallback to nearest upcoming expiry >= today
            today = datetime.now().date()
            matching_upcoming = [
                i for i in instruments
                if i["name"] == "NIFTY" 
                and i["instrument_type"] == opt_type
                and abs(float(i["strike"]) - float(strike)) < 1.0
                and i["expiry"] >= today
            ]
            matching = sorted(matching_upcoming, key=lambda x: x["expiry"])
            if not matching:
                matching = [
                    i for i in instruments
                    if i["name"] == "NIFTY" 
                    and i["instrument_type"] == opt_type
                    and abs(float(i["strike"]) - float(strike)) < 1.0
                ]

        if matching:
            tradingsymbol = matching[0]["tradingsymbol"]
            key = f"NFO:{tradingsymbol}"
            quote_data = kite_client.quote(key)
            ltp = float(quote_data[key]["last_price"])
            return tradingsymbol, ltp
    except Exception as e:
        logger.warning(f"[OptionLookup] Error fetching quote for {strike} {opt_type}: {e}")

    return None, 0.0
