import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def extract_positional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts 7 scale-invariant technical indicators from daily OHLC candles:
    1. norm_ret: Log returns ln(Close_t / Close_{t-1})
    2. hl_range: Daily high-low range normalized by close
    3. norm_atr: 14-day ATR normalized by close
    4. norm_rsi: 14-day RSI scaled to [-1.0, 1.0]
    5. sma20_dev: Distance to 20-day SMA relative to SMA
    6. sma50_dev: Distance to 50-day SMA relative to SMA
    7. realized_vol: Rolling 20-day annualized realized volatility

    Returns:
        DataFrame containing original timestamps/OHLCV plus the 7 feature columns.
    """
    res = df.copy()

    # 1. Log returns
    close_prev = res["close"].shift(1).replace(0, np.nan)
    res["norm_ret"] = np.log(res["close"] / close_prev).fillna(0.0)

    # 2. Daily HL range normalized by close
    res["hl_range"] = ((res["high"] - res["low"]) / res["close"].replace(0, np.nan)).fillna(0.0)

    # 3. ATR-14 normalized by close
    atr = calculate_atr(res, period=14)
    res["norm_atr"] = (atr / res["close"].replace(0, np.nan)).fillna(0.0)

    # 4. RSI-14 scaled to [-1.0, 1.0]
    rsi = calculate_rsi(res, period=14)
    res["norm_rsi"] = ((rsi - 50.0) / 50.0).fillna(0.0)

    # 5. 20-day SMA deviation
    sma20 = res["close"].rolling(20, min_periods=1).mean()
    res["sma20_dev"] = ((res["close"] - sma20) / sma20).fillna(0.0)

    # 6. 50-day SMA deviation
    sma50 = res["close"].rolling(50, min_periods=1).mean()
    res["sma50_dev"] = ((res["close"] - sma50) / sma50).fillna(0.0)

    # 7. 20-day annualized realized volatility
    ret_series = res["close"].pct_change()
    realized_vol = ret_series.rolling(20, min_periods=1).std() * np.sqrt(252)
    res["realized_vol"] = realized_vol.bfill().fillna(0.15)  # 15% default baseline if flat

    feature_cols = ["norm_ret", "hl_range", "norm_atr", "norm_rsi", "sma20_dev", "sma50_dev", "realized_vol"]
    res[feature_cols] = res[feature_cols].bfill().fillna(0.0)

    return res
