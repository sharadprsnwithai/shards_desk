import sys
import os
from pathlib import Path
from datetime import time as dtime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))

from nifty_direction_engine.data_loader import calculate_vwap, calculate_atr
from nifty_direction_engine.lstm_predictor import NiftyLSTMModel

CACHE_DIR = CUR_DIR / "data_cache"
MODEL_SAVE_PATH = CUR_DIR / "nifty_direction_engine" / "lstm_nifty_weights.pt"


def prepare_training_dataset(parquet_path: Path, seq_len: int = 150):
    """
    Builds training samples where:
    - Input sequence: candles up to 10:00 AM on day T (looking back up to seq_len 5m bars, approx 2 days).
    - Features per bar: [norm_open, norm_high, norm_low, norm_close, volume_zscore, vwap_dev]
    - Target: return from 10:00 AM to 11:30 AM (close at 11:30 - close at 10:00) / close at 10:00
    """
    print(f"[INFO] Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("timestamps").reset_index(drop=True)

    # If volume is 0 (typical for index feed from Kite), create synthetic volume proxy from high-low range
    if (df["volume"] == 0).all():
        df["volume_proxy"] = (df["high"] - df["low"]) + 1.0
    else:
        df["volume_proxy"] = df["volume"]

    # Calculate global/rolling technicals
    df["atr14"] = calculate_atr(df, period=14).bfill()
    # Intraday cumulative VWAP using volume_proxy
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["cum_vol_price"] = df.groupby(df["timestamps"].dt.date).apply(
        lambda g: (g["typical_price"] * g["volume_proxy"]).cumsum()
    ).reset_index(level=0, drop=True)
    df["cum_vol"] = df.groupby(df["timestamps"].dt.date)["volume_proxy"].cumsum()
    df["vwap"] = (df["cum_vol_price"] / df["cum_vol"]).bfill()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]

    # Rolling volume zscore using volume_proxy
    v_mean = df["volume_proxy"].rolling(20).mean().bfill()
    v_std = df["volume_proxy"].rolling(20).std().replace(0, 1).bfill()
    df["vol_zscore"] = (df["volume_proxy"] - v_mean) / v_std

    unique_days = sorted(list(set(df["timestamps"].dt.date)))
    print(f"[INFO] Found {len(unique_days)} unique trading sessions.")

    X_list = []
    y_list = []

    for idx, day in enumerate(unique_days):
        day_df = df[df["timestamps"].dt.date == day]
        morn = day_df[day_df["timestamps"].dt.time <= dtime(10, 0)]
        target_win = day_df[(day_df["timestamps"].dt.time > dtime(10, 0)) & (day_df["timestamps"].dt.time <= dtime(11, 30))]

        if len(morn) < 5 or len(target_win) < 5:
            continue

        cutoff_idx = morn.index[-1]
        if cutoff_idx < seq_len:
            continue

        # Target: return from 10:00 to 11:30
        spot_10am = float(morn["close"].iloc[-1])
        close_1130 = float(target_win["close"].iloc[-1])
        ret_1130 = (close_1130 - spot_10am) / spot_10am

        # Input slice of length seq_len up to cutoff_idx
        slice_df = df.loc[cutoff_idx - seq_len + 1:cutoff_idx].copy()
        day_open = float(morn["open"].iloc[0])
        atr_val = float(morn["atr14"].iloc[-1])
        if atr_val <= 0:
            atr_val = 1.0

        # Normalized channels
        norm_open = (slice_df["open"] - day_open) / atr_val
        norm_high = (slice_df["high"] - day_open) / atr_val
        norm_low = (slice_df["low"] - day_open) / atr_val
        norm_close = (slice_df["close"] - day_open) / atr_val
        vol_z = slice_df["vol_zscore"].clip(-3, 3)
        vwap_dev = slice_df["vwap_dev"].clip(-0.05, 0.05)

        features = np.column_stack([norm_open, norm_high, norm_low, norm_close, vol_z, vwap_dev])
        if np.isnan(features).any():
            continue

        X_list.append(features)
        y_list.append(ret_1130)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    print(f"[INFO] Generated {len(X)} training samples. Shape of X: {X.shape}, y: {y.shape}")
    return X, y


class NiftyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_lstm_model(epochs: int = 40, batch_size: int = 16, lr: float = 0.001):
    parquet_path = CACHE_DIR / "nifty_5m_6months.parquet"
    if not parquet_path.exists():
        parquet_path = CACHE_DIR / "NSEI_5m_historical.parquet"

    X, y = prepare_training_dataset(parquet_path, seq_len=120)
    if len(X) < 15:
        print("[ERR] Insufficient training samples.")
        return

    # Train / Validation Split (80% / 20% chronological)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    train_loader = DataLoader(NiftyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NiftyDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training on device: {device}")

    model = NiftyLSTMModel(input_dim=6, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.HuberLoss(delta=0.005) # Robust to intraday market outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    print("[INFO] Starting LSTM training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= len(X_train)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_direction = 0
        total_val = len(X_val)

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)
                
                # Direction hit check
                pred_sign = torch.sign(preds)
                true_sign = torch.sign(batch_y)
                correct_direction += (pred_sign == true_sign).sum().item()

        val_loss /= total_val
        val_acc = (correct_direction / total_val) * 100.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Dir Accuracy: {val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch
            }, MODEL_SAVE_PATH)

    print(f"[SUCCESS] Best LSTM model weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_lstm_model(epochs=35, batch_size=16, lr=0.0015)
