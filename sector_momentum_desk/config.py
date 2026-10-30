"""
sector_momentum_desk/config.py
All configuration constants for the Sector Momentum Strategy.
"""
from pathlib import Path

# ── Bias detection ────────────────────────────────────────────────────────────
BIAS_THRESHOLD: float = 0.0015   # ±0.15 % Nifty return triggers directional bias

# ── Entry / Risk ──────────────────────────────────────────────────────────────
SL_BUFFER: float = 0.0005        # 0.05 % beyond pullback candle extreme for SL

# ── Selection ─────────────────────────────────────────────────────────────────
NUM_TOP_STOCKS: int = 3

# ── Session timing (IST, 24-h strings) ───────────────────────────────────────
CANDLE_INTERVAL: str = "5m"
SESSION_START: str = "09:15"
SELECTION_TIME: str = "09:30"
SESSION_END: str = "15:15"

# ── Cache ─────────────────────────────────────────────────────────────────────
DATA_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "data_cache" / "sector_momentum"

# ── Sector indices (yfinance tickers) ────────────────────────────────────────
SECTOR_TICKERS: dict[str, str] = {
    "IT":     "^CNXIT",
    "Bank":   "^NSEBANK",
    "Auto":   "^CNXAUTO",
    "Pharma": "^CNXPHARMA",
    "FMCG":   "^CNXFMCG",
    "Metal":  "^CNXMETAL",
    "Realty": "^CNXREALTY",
    "Energy": "^CNXENERGY",
}

# ── Sector → F&O constituent stocks (yfinance_ticker, nse_symbol) ────────────
SECTOR_STOCKS: dict[str, list[tuple[str, str]]] = {
    "IT": [
        ("TCS.NS",     "TCS"),     ("INFY.NS",    "INFY"),
        ("WIPRO.NS",   "WIPRO"),   ("HCLTECH.NS", "HCLTECH"),
        ("TECHM.NS",   "TECHM"),   ("LTIM.NS",    "LTIM"),
    ],
    "Bank": [
        ("HDFCBANK.NS",  "HDFCBANK"),  ("ICICIBANK.NS",  "ICICIBANK"),
        ("KOTAKBANK.NS", "KOTAKBANK"), ("AXISBANK.NS",   "AXISBANK"),
        ("SBIN.NS",      "SBIN"),      ("BANKBARODA.NS", "BANKBARODA"),
    ],
    "Auto": [
        ("MARUTI.NS",    "MARUTI"),    ("TATAMOTORS.NS", "TATAMOTORS"),
        ("BAJAJ-AUTO.NS","BAJAJ-AUTO"),("HEROMOTOCO.NS", "HEROMOTOCO"),
        ("EICHERMOT.NS", "EICHERMOT"),("M&M.NS",        "M&M"),
    ],
    "Pharma": [
        ("SUNPHARMA.NS",  "SUNPHARMA"),  ("DRREDDY.NS",   "DRREDDY"),
        ("CIPLA.NS",      "CIPLA"),      ("DIVISLAB.NS",  "DIVISLAB"),
        ("AUROPHARMA.NS", "AUROPHARMA"), ("BIOCON.NS",    "BIOCON"),
    ],
    "FMCG": [
        ("HINDUNILVR.NS", "HINDUNILVR"), ("ITC.NS",       "ITC"),
        ("NESTLEIND.NS",  "NESTLEIND"),  ("BRITANNIA.NS", "BRITANNIA"),
        ("DABUR.NS",      "DABUR"),      ("GODREJCP.NS",  "GODREJCP"),
    ],
    "Metal": [
        ("TATASTEEL.NS",  "TATASTEEL"),  ("HINDALCO.NS",  "HINDALCO"),
        ("JSWSTEEL.NS",   "JSWSTEEL"),   ("VEDL.NS",      "VEDL"),
        ("NATIONALUM.NS", "NATIONALUM"), ("NMDC.NS",      "NMDC"),
    ],
    "Realty": [
        ("DLF.NS",        "DLF"),        ("LODHA.NS",      "LODHA"),
        ("OBEROIRLTY.NS", "OBEROIRLTY"), ("PRESTIGE.NS",   "PRESTIGE"),
        ("GODREJPROP.NS", "GODREJPROP"), ("SOBHA.NS",      "SOBHA"),
    ],
    "Energy": [
        ("RELIANCE.NS",  "RELIANCE"), ("ONGC.NS",     "ONGC"),
        ("BPCL.NS",      "BPCL"),     ("IOC.NS",       "IOC"),
        ("NTPC.NS",      "NTPC"),     ("POWERGRID.NS", "POWERGRID"),
    ],
}
