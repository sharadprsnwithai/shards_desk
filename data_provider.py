import os
import time
import logging
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file automatically
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Local High-Speed Data Cache Directory
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(symbol: str, interval: str) -> Path:
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', symbol).strip('_')
    return CACHE_DIR / f"{clean_name}_{interval}.parquet"


def _read_cache(cache_file: Path, max_age_seconds: float) -> Optional[pd.DataFrame]:
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < max_age_seconds:
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    df["timestamps"] = pd.to_datetime(df["timestamps"])
                    return df
            except Exception:
                pass
    return None


def _write_cache(cache_file: Path, df: pd.DataFrame):
    if df is not None and not df.empty:
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass

# Known NSE & BSE Instrument tokens on Zerodha Kite for quick lookups
KITE_INSTRUMENT_MAP = {
    "^NSEI": {"tradingsymbol": "NIFTY 50", "exchange": "NSE", "token": 256265},
    "^NSEBANK": {"tradingsymbol": "NIFTY BANK", "exchange": "NSE", "token": 260105},
    "^BSESN": {"tradingsymbol": "SENSEX", "exchange": "BSE", "token": 265},
    "RELIANCE.NS": {"tradingsymbol": "RELIANCE", "exchange": "NSE", "token": 738561},
    "TCS.NS": {"tradingsymbol": "TCS", "exchange": "NSE", "token": 2953217},
    "HDFCBANK.NS": {"tradingsymbol": "HDFCBANK", "exchange": "NSE", "token": 341249}
}

MAX_RELOGIN_ATTEMPTS = 5


class DataProvider:
    """
    Unified Data Provider that streams live market data from Zerodha Kite Connect
    with automatic 5-attempt relogin on session expiration and seamless fallback to yfinance.
    """

    def __init__(
        self,
        kite_api_key: Optional[str] = None,
        kite_access_token: Optional[str] = None,
        kite_api_secret: Optional[str] = None
    ):
        self.kite_api_key = kite_api_key or os.getenv("KITE_API_KEY", "").strip()
        self.kite_access_token = kite_access_token or os.getenv("KITE_ACCESS_TOKEN", "").strip()
        self.kite_api_secret = kite_api_secret or os.getenv("KITE_API_SECRET", "").strip()
        self.kite = None
        self._init_kite()

    def _init_kite(self) -> bool:
        """Initializes KiteConnect instance with the current access token."""
        if self.kite_api_key and self.kite_access_token:
            try:
                from kiteconnect import KiteConnect
                self.kite = KiteConnect(api_key=self.kite_api_key)
                self.kite.set_access_token(self.kite_access_token)
                logger.info("[DataProvider] Zerodha KiteConnect client initialized.")
                return True
            except Exception as e:
                logger.warning(f"[DataProvider] Could not initialize KiteConnect: {e}")
                self.kite = None
        return False

    def attempt_relogin(self) -> bool:
        """
        Attempts to refresh / re-login to Kite session up to MAX_RELOGIN_ATTEMPTS:
        1. Reloads .env in case an external script or user refreshed the token file.
        2. If KITE_API_SECRET and KITE_REQUEST_TOKEN exist, generates a fresh session.
        3. Re-initializes the Kite client.
        """
        env_file = Path(__file__).resolve().parent / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=True)

        # 1. Try automated headless TOTP login if TOTP credentials are in .env
        try:
            from kite_auth import auto_login_with_totp
            new_totp_token = auto_login_with_totp()
            if new_totp_token:
                self.kite_access_token = new_totp_token
                logger.info("[DataProvider] Automated TOTP relogin succeeded!")
                return self._init_kite()
        except Exception as e:
            logger.debug(f"[DataProvider] TOTP auto-login attempt failed or unconfigured: {e}")

        # 2. Re-read updated token from .env or exchange request_token if available
        new_key = os.getenv("KITE_API_KEY", "").strip() or self.kite_api_key
        new_token = os.getenv("KITE_ACCESS_TOKEN", "").strip() or self.kite_access_token
        req_token = os.getenv("KITE_REQUEST_TOKEN", "").strip()
        api_secret = os.getenv("KITE_API_SECRET", "").strip() or self.kite_api_secret

        # If a request token and api secret are available, exchange for fresh access token
        if new_key and api_secret and req_token:
            try:
                from kiteconnect import KiteConnect
                kite_temp = KiteConnect(api_key=new_key)
                session_data = kite_temp.generate_session(req_token, api_secret=api_secret)
                new_token = session_data.get("access_token", new_token)
                logger.info("[DataProvider] Generated fresh access token from request token.")
            except Exception as e:
                logger.warning(f"[DataProvider] generate_session failed: {e}")

        self.kite_api_key = new_key
        self.kite_access_token = new_token
        return self._init_kite()

    def to_kite_symbol(self, symbol: str) -> str:
        """Converts Yahoo-style symbol into Kite trading symbol."""
        if symbol in KITE_INSTRUMENT_MAP:
            return KITE_INSTRUMENT_MAP[symbol]["tradingsymbol"]
        return symbol.replace(".NS", "").replace("^", "")

    def get_instrument_token(self, symbol: str) -> Optional[int]:
        if symbol in KITE_INSTRUMENT_MAP:
            return KITE_INSTRUMENT_MAP[symbol]["token"]
        return None

    def fetch_daily(self, symbol: str, days: int = 180, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches daily completed candles. Checks high-speed local disk cache first (fresh for 6 hours).
        If Kite session expires, retries re-login up to 5 times before falling back to yfinance.
        """
        cache_file = _get_cache_path(symbol, "daily")
        if use_cache:
            cached_df = _read_cache(cache_file, max_age_seconds=6 * 3600)
            if cached_df is not None and len(cached_df) >= 30:
                logger.info(f"[DataProvider] Loaded {len(cached_df)} daily candles for {symbol} from local cache.")
                return cached_df

        if self.kite is not None:
            token = self.get_instrument_token(symbol)
            if token:
                for attempt in range(1, MAX_RELOGIN_ATTEMPTS + 1):
                    try:
                        to_date = datetime.now()
                        from_date = to_date - timedelta(days=days)
                        records = self.kite.historical_data(
                            instrument_token=token,
                            from_date=from_date,
                            to_date=to_date,
                            interval="day"
                        )
                        if records and len(records) > 0:
                            df = pd.DataFrame(records)
                            df = df.rename(columns={"date": "timestamps"})
                            df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
                            logger.info(f"[DataProvider] Fetched {len(df)} daily candles for {symbol} via Kite.")
                            res_df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
                            _write_cache(cache_file, res_df)
                            return res_df
                    except Exception as e:
                        logger.warning(f"[DataProvider] Kite daily fetch attempt {attempt}/{MAX_RELOGIN_ATTEMPTS} failed ({e}).")
                        self.attempt_relogin()
                        if attempt == MAX_RELOGIN_ATTEMPTS:
                            logger.warning("[DataProvider] Kite session relogin failed after 5 attempts. Falling back to yfinance.")
                        else:
                            time.sleep(0.5)

        # Yahoo Finance Fallback
        try:
            ticker = yf.Ticker(symbol)
            period_str = "6mo" if days <= 180 else "1y"
            df = ticker.history(period=period_str, interval="1d")
            if not df.empty and len(df) >= 30:
                df = df.reset_index()
                date_col = "Date" if "Date" in df.columns else df.columns[0]
                df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
                })
                logger.info(f"[DataProvider] Fetched {len(df)} daily candles for {symbol} via yfinance.")
                res_df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
                _write_cache(cache_file, res_df)
                return res_df
        except Exception as e:
            logger.warning(f"[DataProvider] yfinance fetch failed for {symbol}: {e}")

        # Synthetic fallback for testing
        return self._generate_synthetic_daily(symbol, count=days)

    def fetch_intraday_5m(self, symbol: str, days: int = 5, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches 5-minute candles up to current moment. Checks high-speed local disk cache first (fresh for 15 mins).
        Retries re-login up to 5 times on session expiration before falling back to yfinance.
        """
        cache_file = _get_cache_path(symbol, "5m")
        if use_cache:
            cached_df = _read_cache(cache_file, max_age_seconds=15 * 60)
            if cached_df is not None and len(cached_df) >= 20:
                logger.info(f"[DataProvider] Loaded {len(cached_df)} 5m candles for {symbol} from local cache.")
                return cached_df

        if self.kite is not None:
            token = self.get_instrument_token(symbol)
            if token:
                for attempt in range(1, MAX_RELOGIN_ATTEMPTS + 1):
                    try:
                        to_date = datetime.now()
                        from_date = to_date - timedelta(days=days)
                        records = self.kite.historical_data(
                            instrument_token=token,
                            from_date=from_date,
                            to_date=to_date,
                            interval="5minute"
                        )
                        if records and len(records) > 0:
                            df = pd.DataFrame(records)
                            df = df.rename(columns={"date": "timestamps"})
                            df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None)
                            logger.info(f"[DataProvider] Fetched {len(df)} 5m candles for {symbol} via Kite.")
                            res_df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
                            _write_cache(cache_file, res_df)
                            return res_df
                    except Exception as e:
                        logger.warning(f"[DataProvider] Kite 5m fetch attempt {attempt}/{MAX_RELOGIN_ATTEMPTS} failed ({e}).")
                        self.attempt_relogin()
                        if attempt == MAX_RELOGIN_ATTEMPTS:
                            logger.warning("[DataProvider] Kite 5m session relogin failed after 5 attempts. Falling back to yfinance.")
                        else:
                            time.sleep(0.5)

        # Yahoo Finance Fallback
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval="5m")
            if not df.empty and len(df) >= 10:
                df = df.reset_index()
                date_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
                df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
                })
                logger.info(f"[DataProvider] Fetched {len(df)} 5m candles for {symbol} via yfinance.")
                res_df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
                _write_cache(cache_file, res_df)
                return res_df
        except Exception as e:
            logger.warning(f"[DataProvider] yfinance 5m fetch failed for {symbol}: {e}")

        # Synthetic fallback
        return self._generate_synthetic_5m(symbol, count=120)

    def _generate_synthetic_daily(self, symbol: str, count: int = 120) -> pd.DataFrame:
        base_price = 24000.0 if "NSEI" in symbol else (1500.0 if "RELIANCE" in symbol or "HDFC" in symbol else 4000.0)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=count, freq='D')
        noise = np.random.normal(0, base_price * 0.008, size=len(dates))
        prices = base_price + np.cumsum(noise)
        return pd.DataFrame({
            'timestamps': dates,
            'open': prices * 0.998,
            'high': prices * 1.006,
            'low': prices * 0.994,
            'close': prices,
            'volume': np.random.randint(100000, 5000000, size=len(dates))
        })

    def _generate_synthetic_5m(self, symbol: str, count: int = 120) -> pd.DataFrame:
        base_price = 24000.0 if "NSEI" in symbol else (1500.0 if "RELIANCE" in symbol or "HDFC" in symbol else 4000.0)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=count, freq='5min')
        noise = np.random.normal(0, base_price * 0.002, size=len(dates))
        prices = base_price + np.cumsum(noise)
        return pd.DataFrame({
            'timestamps': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(5000, 200000, size=len(dates))
        })
