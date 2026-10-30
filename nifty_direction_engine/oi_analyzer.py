from typing import Dict
import pandas as pd

def analyze_oi_flow(
    chain_0915: pd.DataFrame, 
    chain_1000: pd.DataFrame, 
    atm_strike: int, 
    step: int = 50
) -> Dict:
    """
    Analyzes Net ATM/Near-OTM Delta OI Ratio (PE vs CE).
    Assumes chain_xxx is a DataFrame with ['strike', 'type', 'oi']
    """
    # Select 3 nearest ITM and 3 nearest OTM (total 6 strikes)
    # ATM strike is roughly current price.
    
    strikes = [atm_strike + i * step for i in range(-3, 4)]
    
    delta_pe_oi = 0
    delta_ce_oi = 0
    
    for strike in strikes:
        # PE OI
        pe_0915 = chain_0915[(chain_0915['strike'] == strike) & (chain_0915['type'] == 'PE')]['oi'].sum()
        pe_1000 = chain_1000[(chain_1000['strike'] == strike) & (chain_1000['type'] == 'PE')]['oi'].sum()
        delta_pe_oi += (pe_1000 - pe_0915)
        
        # CE OI
        ce_0915 = chain_0915[(chain_0915['strike'] == strike) & (chain_0915['type'] == 'CE')]['oi'].sum()
        ce_1000 = chain_1000[(chain_1000['strike'] == strike) & (chain_1000['type'] == 'CE')]['oi'].sum()
        delta_ce_oi += (ce_1000 - ce_0915)
        
    pe_ce_ratio = delta_pe_oi / delta_ce_oi if delta_ce_oi != 0 else 1.0
    
    sentiment = "NEUTRAL"
    if delta_pe_oi >= 1.25 * delta_ce_oi:
        sentiment = "BULLISH_FLOW"
    elif delta_ce_oi >= 1.25 * delta_pe_oi:
        sentiment = "BEARISH_FLOW"
        
    return {
        "delta_pe_oi": delta_pe_oi,
        "delta_ce_oi": delta_ce_oi,
        "pe_ce_ratio": pe_ce_ratio,
        "flow_sentiment": sentiment
    }
