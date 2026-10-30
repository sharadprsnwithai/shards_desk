import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from nifty_direction_engine.signal_engine import SignalEngine

def test_pipeline_simulation():
    print("🧪 Testing Nifty Directional Pipeline...")
    
    # 1. Create Mock Data
    # 09:15 to 10:00 AM (9 bars)
    mock_data = {
        'open': np.linspace(25000, 25050, 9),
        'high': np.linspace(25010, 25060, 9),
        'low': np.linspace(24990, 25040, 9),
        'close': np.linspace(25005, 25055, 9),
        'volume': np.random.randint(1000, 5000, 9)
    }
    df_10am = pd.DataFrame(mock_data)
    
    # Mock Option Chains
    chain_0915 = pd.DataFrame({
        'strike': [25000, 25050, 25100],
        'type': ['PE', 'PE', 'PE'],
        'oi': [1000, 1000, 1000]
    })
    chain_1000 = pd.DataFrame({
        'strike': [25000, 25050, 25100],
        'type': ['PE', 'PE', 'PE'],
        'oi': [2000, 1200, 1000] # Bullish build-up
    })
    
    # 2. Run Engine
    engine = SignalEngine()
    result = engine.generate_signal(df_10am, chain_0915, chain_1000, atm_strike=25050)
    
    print(f"Result: {result}")
    
    # Validation
    assert "final_bias" in result
    print("✅ Pipeline test passed.")

if __name__ == "__main__":
    test_pipeline_simulation()
