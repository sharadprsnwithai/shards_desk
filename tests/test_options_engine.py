import pytest
import numpy as np
import pandas as pd
from options_engine import (
    snap_strike,
    calculate_range_probability,
    determine_option_strategy,
    determine_intraday_option_strategy
)

def test_snap_strike_rules():
    # 50 step for Nifty / TCS
    assert snap_strike(24183.20, step=50, direction="nearest") == 24200
    assert snap_strike(23740.00, step=50, direction="floor") == 23700
    assert snap_strike(24620.00, step=50, direction="ceil") == 24650

    # 20 step for Reliance / HDFC Bank
    assert snap_strike(1342.0, step=20, direction="nearest") == 1340
    assert snap_strike(1311.5, step=20, direction="floor") == 1300
    assert snap_strike(1384.0, step=20, direction="ceil") == 1400

def test_calculate_range_probability():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=90, freq='D')
    # Generate prices with realistic 12% annualized volatility around 24000
    prices = pd.Series(24000 + np.random.normal(0, 100, size=90), index=dates)
    prob = calculate_range_probability(
        current_price=24000.0,
        support=23000.0,
        resistance=25000.0,
        hist_prices=prices,
        horizon_days=30
    )
    assert 50.0 <= prob <= 99.0

def test_strategy_iron_condor_on_high_probability():
    strat = determine_option_strategy(
        current_price=24180.0,
        support=23740.0,
        resistance=24620.0,
        proj_target=24220.0,  # ~0.16% return (Neutral)
        probability=75.0,
        step=50,
        wing_width=200
    )
    assert strat["name"] == "Iron Condor"
    assert len(strat["legs"]) == 4
    # Sell Put at floor (23700), Buy Put hedge (23500)
    # Sell Call at ceil (24650), Buy Call hedge (24850)
    assert any(l["action"] == "SELL" and l["strike"] == 23700 and l["type"] == "PE" for l in strat["legs"])
    assert any(l["action"] == "BUY" and l["strike"] == 23500 and l["type"] == "PE" for l in strat["legs"])
    assert any(l["action"] == "SELL" and l["strike"] == 24650 and l["type"] == "CE" for l in strat["legs"])
    assert any(l["action"] == "BUY" and l["strike"] == 24850 and l["type"] == "CE" for l in strat["legs"])

def test_strategy_bull_put_on_bullish_bias():
    strat = determine_option_strategy(
        current_price=1750.0,
        support=1710.0,
        resistance=1820.0,
        proj_target=1795.0,  # +2.5% return (Bullish)
        probability=72.0,
        step=20,
        wing_width=40
    )
    assert strat["name"] == "Bull Put Credit Spread"
    assert len(strat["legs"]) == 2
    assert any(l["action"] == "SELL" and l["strike"] == 1700 and l["type"] == "PE" for l in strat["legs"])
    assert any(l["action"] == "BUY" and l["strike"] == 1660 and l["type"] == "PE" for l in strat["legs"])

def test_strategy_bear_call_on_bearish_bias():
    strat = determine_option_strategy(
        current_price=1350.0,
        support=1290.0,
        resistance=1370.0,
        proj_target=1310.0,  # -2.96% return (Bearish)
        probability=70.0,
        step=20,
        wing_width=40
    )
    assert strat["name"] == "Bear Call Credit Spread"
    assert len(strat["legs"]) == 2
    assert any(l["action"] == "SELL" and l["strike"] == 1380 and l["type"] == "CE" for l in strat["legs"])
    assert any(l["action"] == "BUY" and l["strike"] == 1420 and l["type"] == "CE" for l in strat["legs"])

def test_intraday_option_strategy():
    # Bullish intraday setup at 10 AM with 4-OTM (200-pt buffer)
    res = determine_intraday_option_strategy(
        current_price=24150.0,
        day_low=24080.0,
        day_high=24280.0,
        proj_close=24260.0,
        step=50,
        otm_steps=4
    )
    assert res["bias"] == "BULLISH"
    assert "Sell 4-OTM Bull Put Spread" in res["action"]
    assert "23950 PE" in res["action"]
    assert len(res["legs"]) == 2
    assert res["legs"][0]["action"] == "SELL" and res["legs"][0]["strike"] == 23950
    assert res["legs"][1]["action"] == "BUY" and res["legs"][1]["strike"] == 23800
