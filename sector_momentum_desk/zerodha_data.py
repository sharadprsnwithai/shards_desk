"""
sector_momentum_desk/zerodha_data.py
Fetches and caches pristine 6-month historical 5-minute and daily data from Zerodha Kite Connect API.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from sector_momentum_desk.config import DATA_CACHE_DIR, SECTOR_STOCKS

logger = logging.getLogger(__name__)

ZERODHA_CACHE_DIR = DATA_CACHE_DIR / "zerodha_6m"
ZERODHA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Pre-mapped Kite tokens for Indices
INDEX_TOKENS: dict[str, int] = {
    "^NSEI": 256265,      # NIFTY 50
    "IT": 259849,         # NIFTY IT
    "Bank": 260105,       # NIFTY BANK
    "Auto": 263433,       # NIFTY AUTO
    "Pharma": 262409,     # NIFTY PHARMA
    "FMCG": 261897,       # NIFTY FMCG
    "Metal": 263689,      # NIFTY METAL
    "Realty": 261129,     # NIFTY REALTY
    "Energy": 261641,     # NIFTY ENERGY
}


def get_kite_client():
    """Initializes and returns an authenticated KiteConnect client."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)

    from kiteconnect import KiteConnect
    api_key = os.getenv("KITE_API_KEY", "").strip()
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    try:
        kite.profile()
        logger.info("[ZerodhaData] Connected to Zerodha Kite.")
        return kite
    except Exception as exc:
        logger.warning(f"[ZerodhaData] Token expired or invalid ({exc}). Trying TOTP auto-login...")
        try:
            from kite_auth import auto_login_with_totp
            new_token = auto_login_with_totp()
            if new_token:
                kite.set_access_token(new_token)
                kite.profile()
                logger.info("[ZerodhaData] Re-authenticated with fresh TOTP token.")
                return kite
        except Exception as e:
            logger.error(f"[ZerodhaData] Auto-login failed: {e}")
        raise RuntimeError("Unable to authenticate with Zerodha Kite Connect.")


def get_all_instrument_tokens(kite) -> dict[str, int]:
    """Resolves all 48 stock symbols to Kite instrument tokens."""
    token_map = dict(INDEX_TOKENS)
    all_stocks = [nse for s_list in SECTOR_STOCKS.values() for _, nse in s_list]

    df_inst = pd.DataFrame(kite.instruments("NSE"))
    eq = df_inst[df_inst["segment"] == "NSE"]

    for sym in all_stocks:
        m = eq[eq["tradingsymbol"] == sym]
        if not m.empty:
            token_map[sym] = int(m.iloc[0]["instrument_token"])
        else:
            logger.warning(f"[ZerodhaData] Could not find token for {sym}")

    return token_map


def fetch_and_cache_zerodha_data(days: int = 180, force: bool = False):
    """
    Downloads `days` of 5m and daily historical data for all 57 instruments
    from Zerodha Kite API and caches as local Parquet files.
    """
    consolidated_5m_file = ZERODHA_CACHE_DIR / f"consolidated_5m_{days}d.parquet"
    consolidated_daily_file = ZERODHA_CACHE_DIR / f"consolidated_daily_{days}d.parquet"

    if not force and consolidated_5m_file.exists() and consolidated_daily_file.exists():
        logger.info("[ZerodhaData] Loading 6-month Zerodha data from cached parquet...")
        df_5m = pd.read_parquet(consolidated_5m_file)
        df_daily = pd.read_parquet(consolidated_daily_file)
        logger.info(f"[ZerodhaData] Loaded 5m shape: {df_5m.shape}, daily shape: {df_daily.shape}")
        return df_5m, df_daily

    kite = get_kite_client()
    tokens = get_all_instrument_tokens(kite)
    logger.info(f"[ZerodhaData] Starting download of {len(tokens)} instruments from Zerodha API ({days} days)...")

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    dict_5m: dict[str, pd.DataFrame] = {}
    dict_daily: dict[str, pd.DataFrame] = {}

    for i, (sym, token) in enumerate(tokens.items(), 1):
        token_5m_file = ZERODHA_CACHE_DIR / f"{sym}_{token}_5m.parquet"
        token_daily_file = ZERODHA_CACHE_DIR / f"{sym}_{token}_daily.parquet"

        # ── 1. Fetch 5-Minute Candles (in 60-day chunks) ─────────────────────
        if not force and token_5m_file.exists():
            df_sym_5m = pd.read_parquet(token_5m_file)
        else:
            chunks = []
            cur_start = from_date
            while cur_start < to_date:
                cur_end = min(cur_start + timedelta(days=60), to_date)
                try:
                    records = kite.historical_data(token, cur_start, cur_end, interval="5minute")
                    if records:
                        chunks.append(pd.DataFrame(records))
                except Exception as exc:
                    logger.warning(f"Error fetching 5m for {sym} ({cur_start.date()} to {cur_end.date()}): {exc}")
                cur_start = cur_end + timedelta(days=1)
                time.sleep(0.35)

            if chunks:
                df_sym_5m = pd.concat(chunks).drop_duplicates(subset=["date"]).reset_index(drop=True)
                df_sym_5m["date"] = pd.to_datetime(df_sym_5m["date"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                df_sym_5m = df_sym_5m.rename(columns={
                    "date": "timestamps", "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume"
                })
                df_sym_5m.to_parquet(token_5m_file)
            else:
                df_sym_5m = pd.DataFrame()

        dict_5m[sym] = df_sym_5m

        # ── 2. Fetch Daily Candles (single request) ───────────────────────────
        if not force and token_daily_file.exists():
            df_sym_daily = pd.read_parquet(token_daily_file)
        else:
            try:
                rec_daily = kite.historical_data(token, from_date - timedelta(days=30), to_date, interval="day")
                if rec_daily:
                    df_sym_daily = pd.DataFrame(rec_daily).drop_duplicates(subset=["date"]).reset_index(drop=True)
                    df_sym_daily["date"] = pd.to_datetime(df_sym_daily["date"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                    df_sym_daily = df_sym_daily.rename(columns={
                        "date": "timestamps", "open": "Open", "high": "High",
                        "low": "Low", "close": "Close", "volume": "Volume"
                    })
                    df_sym_daily.to_parquet(token_daily_file)
                else:
                    df_sym_daily = pd.DataFrame()
            except Exception as exc:
                logger.warning(f"Error fetching daily for {sym}: {exc}")
                df_sym_daily = pd.DataFrame()
            time.sleep(0.35)

        dict_daily[sym] = df_sym_daily
        logger.info(f"[{i}/{len(tokens)}] {sym:<12} (token {token}): {len(df_sym_5m)} 5m bars, {len(df_sym_daily)} daily bars")

    # ── 3. Build Multi-Index Consolidated DataFrames ─────────────────────────
    # Format matching backtester expectations: index=timestamps, columns=(Field, Symbol)
    all_timestamps_5m = sorted(list(set(ts for df in dict_5m.values() if not df.empty for ts in df["timestamps"])))
    idx_5m = pd.DatetimeIndex(all_timestamps_5m)

    data_5m_dict = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        col_df = pd.DataFrame(index=idx_5m)
        for sym, df in dict_5m.items():
            if not df.empty:
                s = df.set_index("timestamps")[field]
                col_df[sym] = s
        data_5m_dict[field] = col_df

    df_5m_multi = pd.concat(data_5m_dict, axis=1)
    df_5m_multi.to_parquet(consolidated_5m_file)

    all_timestamps_daily = sorted(list(set(ts for df in dict_daily.values() if not df.empty for ts in df["timestamps"])))
    idx_daily = pd.DatetimeIndex(all_timestamps_daily)

    data_daily_dict = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        col_df = pd.DataFrame(index=idx_daily)
        for sym, df in dict_daily.items():
            if not df.empty:
                s = df.set_index("timestamps")[field]
                col_df[sym] = s
        data_daily_dict[field] = col_df

    df_daily_multi = pd.concat(data_daily_dict, axis=1)
    df_daily_multi.to_parquet(consolidated_daily_file)

    logger.info(f"[ZerodhaData] Consolidated 6-month data saved to {consolidated_5m_file}")
    return df_5m_multi, df_daily_multi
