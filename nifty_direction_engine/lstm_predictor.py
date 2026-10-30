import sys
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class NiftyLSTMModel(nn.Module):
    """
    Bidirectional LSTM with self-attention / dense projection
    for predicting 10:00 -> 11:30 AM return from prior multi-day 5m sequence.
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, 1) # Outputs predicted return %

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        out, (h_n, c_n) = self.lstm(x)
        # Take the last time step output
        last_hidden = out[:, -1, :]
        feat = self.dropout(self.relu(self.fc1(last_hidden)))
        pred_return = self.fc_out(feat)
        return pred_return

class LSTMDirectionPredictor:
    """
    Wrapper for inference that conforms to the predictor interface.
    """
    def __init__(self, weights_path: str = None, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = NiftyLSTMModel(input_dim=6, hidden_dim=64, num_layers=2).to(self.device)
        if weights_path and Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.mean = checkpoint.get("mean", None)
            self.std = checkpoint.get("std", None)
            self.model.eval()
            print(f"[LSTMDirectionPredictor] Loaded model from {weights_path}")
        else:
            self.mean = None
            self.std = None
            self.model.eval()

    def get_prediction(self, feature_tensor: torch.Tensor) -> float:
        """
        Takes input tensor of shape [1, seq_len, 6] or numpy array and returns predicted return.
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(feature_tensor, np.ndarray):
                feature_tensor = torch.tensor(feature_tensor, dtype=torch.float32)
            if feature_tensor.dim() == 2:
                feature_tensor = feature_tensor.unsqueeze(0)
            feature_tensor = feature_tensor.to(self.device)
            pred = self.model(feature_tensor)
            return float(pred.squeeze().item())
