from nifty_direction_engine.data_loader import preprocess_features, calculate_vwap
from nifty_direction_engine.kronos_predictor import KronosDirectionPredictor
from nifty_direction_engine.oi_analyzer import analyze_oi_flow

class SignalEngine:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        if not use_mock:
            self.predictor = KronosDirectionPredictor()
        
    def generate_signal(self, df_10am, chain_0915, chain_1000, atm_strike):
        # 1. Prediction (Mock or Real)
        if self.use_mock:
            # Randomly simulate a prediction return
            import numpy as np
            pred_return = np.random.uniform(-0.01, 0.01)
        else:
            norm_df = preprocess_features(df_10am)
            pred_return, _ = self.predictor.get_prediction(norm_df[['open', 'high', 'low', 'close']])
        
        # 2. Gating
        # Rule 1: Structural Deadband
        if abs(pred_return) < 0.0025:
            return {"final_bias": "SIDEWAYS_NO_TRADE", "reason": "Structural Deadband"}
            
        # Rule 2: VWAP Acceptance
        vwap = calculate_vwap(df_10am).iloc[-1]
        price_10am = df_10am['close'].iloc[-1]
        
        if pred_return > 0 and price_10am < vwap:
            return {"final_bias": "SIDEWAYS_NO_TRADE", "reason": "VWAP Rejection (Bullish Prediction but below VWAP)"}
        if pred_return < 0 and price_10am > vwap:
            return {"final_bias": "SIDEWAYS_NO_TRADE", "reason": "VWAP Rejection (Bearish Prediction but above VWAP)"}
            
        # Rule 3: OI Flow
        oi_analysis = analyze_oi_flow(chain_0915, chain_1000, atm_strike)
        
        if pred_return > 0 and oi_analysis['flow_sentiment'] != "BULLISH_FLOW":
            return {"final_bias": "SIDEWAYS_NO_TRADE", "reason": "OI Flow Mismatch (Bullish)"}
        if pred_return < 0 and oi_analysis['flow_sentiment'] != "BEARISH_FLOW":
            return {"final_bias": "SIDEWAYS_NO_TRADE", "reason": "OI Flow Mismatch (Bearish)"}
            
        return {
            "final_bias": "BULLISH" if pred_return > 0 else "BEARISH",
            "confidence": "HIGH",
            "pred_return": pred_return
        }
