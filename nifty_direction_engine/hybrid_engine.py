import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

CUR_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CUR_DIR))

from nifty_direction_engine.data_loader import preprocess_features, calculate_vwap, calculate_atr
from nifty_direction_engine.kronos_predictor import KronosDirectionPredictor
from nifty_direction_engine.lstm_predictor import LSTMDirectionPredictor
from nifty_direction_engine.oi_analyzer import analyze_oi_flow

WEIGHTS_PATH = Path(__file__).resolve().parent / "lstm_nifty_weights.pt"

class HybridSignalEngine:
    """
    Ensemble Intraday Directional Engine:
    - Kronos (Time-Series Transformer): captures range expansion & multi-day regime
    - LSTM (Locally trained PyTorch network): calibrated on 5m Nifty microstructure
    - Deterministic Gating Rules:
        1. Structural Deadband: |r_hat| >= 0.25% (or 0.20% minimum threshold)
        2. VWAP Acceptance: 10:00 AM Spot vs VWAP
        3. Derivatives Flow Gate: Delta PE vs Delta CE OI ratio (>= 1.25x)
    """

    def __init__(self, use_kronos: bool = True, use_lstm: bool = True, weights_path: str = None):
        self.use_kronos = use_kronos
        self.use_lstm = use_lstm

        if self.use_kronos:
            print("[HybridSignalEngine] Initializing Kronos Transformer...")
            self.kronos = KronosDirectionPredictor()
        else:
            self.kronos = None

        if self.use_lstm:
            path = weights_path or str(WEIGHTS_PATH)
            print(f"[HybridSignalEngine] Initializing LSTM Predictor from {path}...")
            try:
                self.lstm = LSTMDirectionPredictor(weights_path=path)
                self._lstm_error: str = ""
            except RuntimeError as e:
                self.lstm = None
                self._lstm_error = str(e)
                print(f"[HybridSignalEngine] ⛔ LSTM unavailable: {e}")
        else:
            self.lstm = None
            self._lstm_error = ""

    def _prepare_lstm_input(self, df_lookback: pd.DataFrame, day_open: float, atr_val: float) -> np.ndarray:
        """Constructs [1, seq_len, 6] feature tensor for the LSTM."""
        if atr_val <= 0:
            atr_val = 1.0

        norm_open = (df_lookback["open"] - day_open) / atr_val
        norm_high = (df_lookback["high"] - day_open) / atr_val
        norm_low = (df_lookback["low"] - day_open) / atr_val
        norm_close = (df_lookback["close"] - day_open) / atr_val

        # Volume proxy
        vol = df_lookback["volume"]
        if (vol == 0).all():
            vol = (df_lookback["high"] - df_lookback["low"]) + 1.0

        v_mean = vol.rolling(20, min_periods=1).mean()
        v_std = vol.rolling(20, min_periods=1).std().replace(0, 1).fillna(1)
        vol_z = ((vol - v_mean) / v_std).clip(-3, 3)

        # VWAP deviation
        vwap = calculate_vwap(df_lookback).bfill()
        vwap_dev = ((df_lookback["close"] - vwap) / vwap).clip(-0.05, 0.05)

        feat_matrix = np.column_stack([norm_open, norm_high, norm_low, norm_close, vol_z, vwap_dev])
        return np.nan_to_num(feat_matrix, nan=0.0)

    def generate_signal(
        self,
        df_lookback: pd.DataFrame,
        chain_0915: pd.DataFrame = None,
        chain_1000: pd.DataFrame = None,
        atm_strike: int = None,
        deadband: float = 0.0020
    ) -> dict:
        """
        Input:
            df_lookback: 5m candles up to 10:00 AM (at least 75-120 bars recommended)
            chain_0915, chain_1000: Option chain snapshots for OI flow analysis
            atm_strike: Nearest ATM strike for derivative flow
        """
        spot_10am = float(df_lookback["close"].iloc[-1])
        day_open = float(df_lookback["open"].iloc[0])
        atr_val = float(calculate_atr(df_lookback).dropna().iloc[-1]) if len(df_lookback) >= 14 else 50.0

        # Guard: if LSTM was requested but weights are missing, refuse to trade
        if self.use_lstm and self.lstm is None and self._lstm_error:
            return {
                "final_bias": "SIDEWAYS_NO_TRADE",
                "reason": f"Model weights unavailable: {self._lstm_error}"
            }

        predictions = []
        kronos_ret = 0.0
        lstm_ret = 0.0

        # 1. Kronos Prediction
        if self.use_kronos and self.kronos is not None:
            try:
                x_ts = df_lookback["timestamps"] if "timestamps" in df_lookback else None
                kronos_ret, _ = self.kronos.get_prediction(df_lookback, x_timestamp=x_ts, pred_len=18)
                predictions.append(kronos_ret)
            except Exception as e:
                print(f"[WARN] Kronos prediction failed: {e}")

        # 2. LSTM Prediction
        if self.use_lstm and self.lstm is not None:
            try:
                lstm_feats = self._prepare_lstm_input(df_lookback, day_open, atr_val)
                lstm_ret = self.lstm.get_prediction(lstm_feats)
                predictions.append(lstm_ret)
            except Exception as e:
                print(f"[WARN] LSTM prediction failed: {e}")

        if not predictions:
            return {"final_bias": "SIDEWAYS_NO_TRADE", "reason": "No model output available"}

        # Ensemble Return (Average)
        ensemble_ret = sum(predictions) / len(predictions)

        # ------------------------------------------------------------------
        # Deterministic Gating Rules (from Engineering Specification)
        # ------------------------------------------------------------------

        # Gate 1: Structural Deadband
        if abs(ensemble_ret) < deadband:
            return {
                "final_bias": "SIDEWAYS_NO_TRADE",
                "reason": f"Deadband Filter (|{ensemble_ret*100:.2f}%| < {deadband*100:.2f}%)",
                "pred_return": ensemble_ret,
                "kronos_ret": kronos_ret,
                "lstm_ret": lstm_ret,
                "spot_10am": spot_10am
            }

        # Gate 2: VWAP Acceptance
        vwap_series = calculate_vwap(df_lookback)
        vwap_10am = float(vwap_series.iloc[-1])

        if ensemble_ret > 0 and spot_10am < vwap_10am:
            return {
                "final_bias": "SIDEWAYS_NO_TRADE",
                "reason": f"VWAP Rejection (Bullish {ensemble_ret*100:+.2f}% but Spot {spot_10am:.1f} < VWAP {vwap_10am:.1f})",
                "pred_return": ensemble_ret,
                "kronos_ret": kronos_ret,
                "lstm_ret": lstm_ret,
                "spot_10am": spot_10am
            }

        if ensemble_ret < 0 and spot_10am > vwap_10am:
            return {
                "final_bias": "SIDEWAYS_NO_TRADE",
                "reason": f"VWAP Rejection (Bearish {ensemble_ret*100:+.2f}% but Spot {spot_10am:.1f} > VWAP {vwap_10am:.1f})",
                "pred_return": ensemble_ret,
                "kronos_ret": kronos_ret,
                "lstm_ret": lstm_ret,
                "spot_10am": spot_10am
            }

        # Gate 3: Derivatives Flow Gate (when option chain is available)
        oi_flow = "NEUTRAL"
        if chain_0915 is not None and chain_1000 is not None and atm_strike is not None:
            oi_res = analyze_oi_flow(chain_0915, chain_1000, atm_strike)
            oi_flow = oi_res.get("flow_sentiment", "NEUTRAL")

            if ensemble_ret > 0 and oi_flow == "BEARISH_FLOW":
                return {
                    "final_bias": "SIDEWAYS_NO_TRADE",
                    "reason": "Flow Gate Rejection (Bullish model vs Bearish Call Writing resistance)",
                    "pred_return": ensemble_ret,
                    "kronos_ret": kronos_ret,
                    "lstm_ret": lstm_ret,
                    "spot_10am": spot_10am
                }
            if ensemble_ret < 0 and oi_flow == "BULLISH_FLOW":
                return {
                    "final_bias": "SIDEWAYS_NO_TRADE",
                    "reason": "Flow Gate Rejection (Bearish model vs Bullish Put Writing floor)",
                    "pred_return": ensemble_ret,
                    "kronos_ret": kronos_ret,
                    "lstm_ret": lstm_ret,
                    "spot_10am": spot_10am
                }

        final_bias = "BULLISH" if ensemble_ret > 0 else "BEARISH"
        return {
            "final_bias": final_bias,
            "confidence": "HIGH" if (kronos_ret * lstm_ret > 0) else "MODERATE",
            "reason": "All Gates Passed",
            "pred_return": ensemble_ret,
            "kronos_ret": kronos_ret,
            "lstm_ret": lstm_ret,
            "spot_10am": spot_10am,
            "vwap_10am": vwap_10am,
            "oi_flow": oi_flow
        }
