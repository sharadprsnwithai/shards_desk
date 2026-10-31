import math
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import norm


def snap_strike(price: float, step: int, direction: str = "nearest") -> int:
    """
    Snaps a floating price to official exchange-traded option strike step sizes.

    Parameters:
        price: Price level (e.g. 24183.20)
        step: Tick size / strike interval (e.g. 50 for Nifty, 20 for Reliance)
        direction: 'nearest', 'floor' (at or below), or 'ceil' (at or above)
    """
    if step <= 0:
        return int(round(price))

    if direction == "floor":
        return int(math.floor(price / step) * step)
    elif direction == "ceil":
        return int(math.ceil(price / step) * step)
    else:  # nearest
        return int(round(price / step) * step)


def calculate_range_probability(
    current_price: float,
    support: float,
    resistance: float,
    hist_prices: pd.Series,
    horizon_days: int = 30
) -> float:
    """
    Calculates the analytical probability of the underlying asset staying strictly
    between support and resistance boundaries over the given horizon, using the
    asset's annualized historical realized volatility diffusion cone.
    """
    if len(hist_prices) < 5 or current_price <= 0 or support <= 0 or resistance <= support:
        return 65.0  # Safe default baseline

    # Calculate log returns and annualized realized volatility
    log_returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
    daily_vol = float(log_returns.std())
    annual_vol = daily_vol * np.sqrt(252)

    # Limit extreme volatility bounds (e.g., between 8% and 60%)
    sigma = max(0.08, min(annual_vol, 0.60))
    t = horizon_days / 365.0
    denom = sigma * np.sqrt(t)

    # For range bounds, evaluate standard diffusion without arbitrary directional drift
    d2_upper = np.log(resistance / current_price) / denom
    d2_lower = np.log(support / current_price) / denom

    p_upper = norm.cdf(d2_upper)
    p_lower = norm.cdf(d2_lower)

    prob = float((p_upper - p_lower) * 100.0)
    # Clamp between 20% and 95%
    return round(max(20.0, min(prob, 95.0)), 1)


def determine_option_strategy(
    current_price: float,
    support: float,
    resistance: float,
    proj_target: float,
    probability: float,
    step: int,
    wing_width: int
) -> Dict[str, Any]:
    """
    Determines the optimal quantitative option selling strategy and exact strike legs
    based on range bounds, directional momentum, and containment probability.
    """
    ret_pct = ((proj_target - current_price) / current_price) * 100.0

    if ret_pct >= 1.0:
        bias = "Bullish"
    elif ret_pct <= -1.0:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # Rule Matrix
    if probability < 45.0:
        return {
            "name": "NO TRADE (Extreme Volatility / Breakout Risk)",
            "bias": bias,
            "expected_return_pct": round(ret_pct, 2),
            "legs": []
        }

    # Case 1: High Range Probability & Neutral Trend -> IRON CONDOR
    if probability >= 60.0 and bias == "Neutral":
        sell_put = snap_strike(support, step, direction="floor")
        buy_put = sell_put - wing_width
        sell_call = snap_strike(resistance, step, direction="ceil")
        buy_call = sell_call + wing_width

        legs = [
            {"action": "SELL", "strike": sell_put, "type": "PE", "desc": "Put Leg @ Support"},
            {"action": "BUY",  "strike": buy_put,  "type": "PE", "desc": f"Put Wing Hedge (-{wing_width})"},
            {"action": "SELL", "strike": sell_call, "type": "CE", "desc": "Call Leg @ Resistance"},
            {"action": "BUY",  "strike": buy_call,  "type": "CE", "desc": f"Call Wing Hedge (+{wing_width})"}
        ]
        return {
            "name": "Iron Condor",
            "bias": bias,
            "expected_return_pct": round(ret_pct, 2),
            "legs": legs
        }

    # Case 2: Bullish Momentum -> BULL PUT CREDIT SPREAD
    if bias == "Bullish" and probability >= 50.0:
        sell_put = snap_strike(min(support, current_price * 0.985), step, direction="floor")
        buy_put = sell_put - wing_width
        legs = [
            {"action": "SELL", "strike": sell_put, "type": "PE", "desc": "Sell Put @ Strong Support Floor"},
            {"action": "BUY",  "strike": buy_put,  "type": "PE", "desc": f"Protective Put Wing Hedge (-{wing_width})"}
        ]
        return {
            "name": "Bull Put Credit Spread",
            "bias": bias,
            "expected_return_pct": round(ret_pct, 2),
            "legs": legs
        }

    # Case 3: Bearish Momentum -> BEAR CALL CREDIT SPREAD
    if bias == "Bearish" and probability >= 50.0:
        sell_call = snap_strike(max(resistance, current_price * 1.015), step, direction="ceil")
        buy_call = sell_call + wing_width
        legs = [
            {"action": "SELL", "strike": sell_call, "type": "CE", "desc": "Sell Call @ Resistance Ceiling"},
            {"action": "BUY",  "strike": buy_call,  "type": "CE", "desc": f"Protective Call Wing Hedge (+{wing_width})"}
        ]
        return {
            "name": "Bear Call Credit Spread",
            "bias": bias,
            "expected_return_pct": round(ret_pct, 2),
            "legs": legs
        }

    # Fallback to wider Iron Condor
    sell_put = snap_strike(support, step, direction="floor")
    buy_put = sell_put - wing_width
    sell_call = snap_strike(resistance, step, direction="ceil")
    buy_call = sell_call + wing_width
    return {
        "name": "Iron Condor (Wide Strangle Wing)",
        "bias": bias,
        "expected_return_pct": round(ret_pct, 2),
        "legs": [
            {"action": "SELL", "strike": sell_put, "type": "PE", "desc": "Support Leg"},
            {"action": "BUY",  "strike": buy_put,  "type": "PE", "desc": "Hedge"},
            {"action": "SELL", "strike": sell_call, "type": "CE", "desc": "Resistance Leg"},
            {"action": "BUY",  "strike": buy_call,  "type": "CE", "desc": "Hedge"}
        ]
    }


def determine_intraday_option_strategy(
    current_price: float,
    day_low: float,
    day_high: float,
    proj_close: float,
    step: int = 50,
    otm_steps: int = 4
) -> Dict[str, Any]:
    """
    Determines 10:00 AM same-day option selling setups using the NIFTY 4-OTM (200-pt buffer) rule.
    - Strike Distance: 4 strikes OTM = 200 points away from spot.
    - Profit Booking: 80% premium decay.
    - Stop-Loss: 50% premium expansion.
    """
    move_pct = ((proj_close - current_price) / current_price) * 100.0
    otm_distance = otm_steps * step  # 4 * 50 = 200 points for Nifty

    safe_put = snap_strike(min(day_low, current_price - otm_distance), step, direction="floor")
    safe_call = snap_strike(max(day_high, current_price + otm_distance), step, direction="ceil")

    if move_pct >= 0.20:
        bias = "BULLISH"
        legs = [
            {"action": "SELL", "strike": safe_put, "type": "PE", "desc": "4-OTM Main Leg"},
            {"action": "BUY",  "strike": safe_put - 150, "type": "PE", "desc": "Hedge Wing (Margin Saver)"}
        ]
        action = f"Sell 4-OTM Bull Put Spread: Sell {safe_put} PE | Buy {safe_put - 150} PE"
    elif move_pct <= -0.20:
        bias = "BEARISH"
        legs = [
            {"action": "SELL", "strike": safe_call, "type": "CE", "desc": "4-OTM Main Leg"},
            {"action": "BUY",  "strike": safe_call + 150, "type": "CE", "desc": "Hedge Wing (Margin Saver)"}
        ]
        action = f"Sell 4-OTM Bear Call Spread: Sell {safe_call} CE | Buy {safe_call + 150} CE"
    else:
        bias = "NEUTRAL"
        legs = [
            {"action": "SELL", "strike": safe_put, "type": "PE", "desc": "4-OTM Put Leg"},
            {"action": "BUY",  "strike": safe_put - 150, "type": "PE", "desc": "Put Hedge Wing"},
            {"action": "SELL", "strike": safe_call, "type": "CE", "desc": "4-OTM Call Leg"},
            {"action": "BUY",  "strike": safe_call + 150, "type": "CE", "desc": "Call Hedge Wing"}
        ]
        action = f"Sell 4-OTM Iron Condor: {safe_put}P/{safe_call}C"

    return {
        "bias": bias,
        "move_pct": round(move_pct, 2),
        "action": action,
        "legs": legs,
        "safe_put": safe_put,
        "safe_call": safe_call,
        "otm_distance": otm_distance
    }
