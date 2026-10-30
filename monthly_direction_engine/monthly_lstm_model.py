import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from monthly_direction_engine.feature_extractor import extract_positional_features

logger = logging.getLogger(__name__)

FEATURE_COLS = ["norm_ret", "hl_range", "norm_atr", "norm_rsi", "sma20_dev", "sma50_dev", "realized_vol"]


class MonthlyLSTMModel(nn.Module):
    """
    Scale-invariant Positional LSTM for predicting 30-day forward return
    from historical daily multi-asset technical features.
    """
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, input_dim]
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        feat = self.dropout_layer(self.relu(self.fc1(last_hidden)))
        pred_return = self.fc_out(feat)
        return pred_return


class MonthlyLSTMPredictor:
    """
    Inference wrapper for the 30-day positional LSTM model.
    Handles scale normalization, device allocation, and graceful defaults.
    """
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = MonthlyLSTMModel(input_dim=len(FEATURE_COLS), hidden_dim=64, num_layers=2)
        self.model.to(self.device)
        self.mean = None
        self.std = None
        self.is_loaded = False

        if weights_path:
            p = Path(weights_path)
            if p.exists():
                try:
                    checkpoint = torch.load(p, map_location=self.device)
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.mean = checkpoint.get("mean", None)
                    self.std = checkpoint.get("std", None)
                    self.is_loaded = True
                    logger.info(f"[MonthlyLSTM] Loaded weights from {weights_path}")
                except Exception as e:
                    logger.warning(f"[MonthlyLSTM] Failed loading weights from {weights_path}: {e}")

        self.model.eval()

    def predict_return(self, df: pd.DataFrame, seq_len: int = 60) -> float:
        """
        Extracts positional features and forecasts the 30-day forward return percentage (as a decimal fraction).
        E.g., +0.025 means +2.5% projected return.
        """
        if df is None or len(df) < seq_len:
            return 0.0

        try:
            feats_df = extract_positional_features(df)
            sub = feats_df.iloc[-seq_len:][FEATURE_COLS].copy()
            arr = sub.to_numpy(dtype=np.float32)  # [seq_len, 7]

            if self.mean is not None and self.std is not None:
                mean_arr = np.array(self.mean, dtype=np.float32)
                std_arr = np.array(self.std, dtype=np.float32)
                std_arr[std_arr == 0] = 1.0
                arr = (arr - mean_arr) / std_arr

            tensor_in = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(tensor_in).squeeze().item()
            return float(pred)
        except Exception as e:
            logger.warning(f"[MonthlyLSTM] Prediction failed: {e}. Returning neutral 0.0")
            return 0.0
