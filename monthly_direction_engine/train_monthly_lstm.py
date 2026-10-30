import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CUR_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CUR_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from data_provider import DataProvider
from monthly_direction_engine.feature_extractor import extract_positional_features
from monthly_direction_engine.monthly_lstm_model import MonthlyLSTMModel, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_ASSETS = ["^NSEI", "^NSEBANK", "^BSESN", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
MODEL_SAVE_PATH = CUR_DIR / "monthly_lstm_weights.pt"


def build_pooled_dataset(
    symbols: List[str] = TARGET_ASSETS,
    seq_len: int = 60,
    pred_horizon: int = 30,
    days: int = 1800
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downloads historical daily data for all symbols and builds pooled sliding window samples.
    """
    dp = DataProvider()
    all_x = []
    all_y = []

    for sym in symbols:
        logger.info(f"[Training] Fetching daily history for {sym}...")
        df = dp.fetch_daily(sym, days=days, use_cache=True)
        if df is None or len(df) < (seq_len + pred_horizon + 10):
            logger.warning(f"[Training] Insufficient daily data for {sym} ({len(df) if df is not None else 0} bars). Skipping.")
            continue

        feats_df = extract_positional_features(df)
        feat_matrix = feats_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        closes = feats_df["close"].to_numpy(dtype=np.float32)

        n_bars = len(feat_matrix)
        sym_x = []
        sym_y = []

        for i in range(seq_len, n_bars - pred_horizon):
            window_x = feat_matrix[i - seq_len : i]
            # Forward 30-day return fraction
            curr_c = closes[i - 1]
            future_c = closes[i + pred_horizon - 1]
            target_ret = (future_c - curr_c) / (curr_c + 1e-8)

            sym_x.append(window_x)
            sym_y.append(target_ret)

        logger.info(f"[Training] Extracted {len(sym_x)} samples for {sym}.")
        all_x.extend(sym_x)
        all_y.extend(sym_y)

    if not all_x:
        raise ValueError("No training samples could be extracted across the target assets.")

    X = np.array(all_x, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32).reshape(-1, 1)

    logger.info(f"[Training] Total pooled samples: {len(X)} across {len(symbols)} assets.")

    # Calculate global scale standardization across sequence and feature dimensions
    # Shape: [N, seq_len, 7] -> mean & std over (0, 1)
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    std[std == 0] = 1.0

    X_norm = (X - mean) / std

    return X_norm, y, mean, std


def train_multi_asset_monthly_lstm(
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.001,
    seq_len: int = 60,
    pred_horizon: int = 30
) -> Path:
    """
    Trains the multi-asset Positional LSTM model and saves checkpoint with weights, mean, and std.
    """
    X_norm, y, mean, std = build_pooled_dataset(seq_len=seq_len, pred_horizon=pred_horizon)

    # Train / Validation Split (85% Train, 15% Val)
    n_total = len(X_norm)
    indices = np.random.RandomState(42).permutation(n_total)
    split_idx = int(n_total * 0.85)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_ds = TensorDataset(torch.tensor(X_norm[train_idx]), torch.tensor(y[train_idx]))
    val_ds = TensorDataset(torch.tensor(X_norm[val_idx]), torch.tensor(y[val_idx]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Training] Starting training on device: {device} ({len(train_ds)} train, {len(val_ds)} val)...")

    model = MonthlyLSTMModel(input_dim=len(FEATURE_COLS), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.SmoothL1Loss()  # Robust to market return outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} (Best: {best_val_loss:.5f})")

    # Save checkpoint
    torch.save({
        "model_state_dict": best_state,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_cols": FEATURE_COLS,
        "seq_len": seq_len,
        "pred_horizon": pred_horizon,
        "best_val_loss": best_val_loss
    }, MODEL_SAVE_PATH)

    logger.info(f"[Training] Model successfully trained and saved to {MODEL_SAVE_PATH} (size: {MODEL_SAVE_PATH.stat().st_size / 1024:.1f} KB)")
    return MODEL_SAVE_PATH


if __name__ == "__main__":
    train_multi_asset_monthly_lstm()
