import math
import numpy as np
import pandas as pd

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculates the cumulative VWAP for a 5-minute dataframe."""
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates ATR for a 5-minute dataframe."""
    high_low = df['high'] - df['low']
    high_prev_close = (df['high'] - df['close'].shift(1)).abs()
    low_prev_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies open-anchored normalization and calculates feature channels.
    Expects df with ['open', 'high', 'low', 'close', 'volume']
    """
    df = df.copy()
    day_open = df['open'].iloc[0] # Assuming anchored to start of data segment
    atr = calculate_atr(df)
    vwap = calculate_vwap(df)
    
    # Normalization
    df['norm_open'] = (df['open'] - day_open) / atr
    df['norm_high'] = (df['high'] - day_open) / atr
    df['norm_low'] = (df['low'] - day_open) / atr
    df['norm_close'] = (df['close'] - day_open) / atr
    
    # Volume Z-Score
    vol_mean = df['volume'].rolling(20).mean()
    vol_std = df['volume'].rolling(20).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std
    
    # VWAP Deviation
    df['vwap_dev'] = (df['close'] - vwap) / vwap
    
    return df.fillna(0)
