import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, Tuple
from data_provider import DataProvider

logger = logging.getLogger(__name__)

def get_current_tuesday_expiry(kite_client) -> Optional[date]:
    """
    Finds the closest upcoming Tuesday expiry for NIFTY options.
    If today is Tuesday, it returns today's expiry.
    """
    try:
        instruments = kite_client.instruments("NFO")
        tuesday_expiries = sorted(list(set(
            i["expiry"] for i in instruments 
            if i["name"] == "NIFTY" and i["instrument_type"] in ["CE", "PE"] and i["expiry"].weekday() == 1
        )))
        
        today = datetime.now().date()
        valid_expiries = [e for e in tuesday_expiries if e >= today]
        if valid_expiries:
            return valid_expiries[0]
        elif tuesday_expiries:
            return tuesday_expiries[-1]
    except Exception as e:
        logger.warning(f"[OptionLookup] Failed to find Tuesday expiry: {e}")
        
    # Fallback to computing closest Tuesday date
    today = datetime.now().date()
    days_ahead = (1 - today.weekday()) % 7 # 1 = Tuesday
    return today + timedelta(days=days_ahead)


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
            # Fallback without strict expiry
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
