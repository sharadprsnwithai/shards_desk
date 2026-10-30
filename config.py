import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env from project root if present
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class Config:
    kite_api_key: str
    kite_access_token: str
    telegram_bot_token: str
    telegram_chat_id: str
    tz: str
    use_yfinance_fallback: bool
    intraday_option_selling: str = "PAPER"
    trade_lots: int = 1
    auto_trade: bool = True
    product_type: str = "MIS"

    @classmethod
    def load(cls) -> "Config":
        # Primary check: INTRADAY_OPTION_SELLING (PAPER or LIVE)
        mode = os.getenv("INTRADAY_OPTION_SELLING", "").strip().upper()
        if not mode:
            mode = os.getenv("TRADING_MODE", "PAPER").strip().upper()

        if mode not in ("LIVE", "PAPER"):
            mode = "PAPER"

        try:
            lots = max(1, int(os.getenv("TRADE_LOTS", "1")))
        except ValueError:
            lots = 1

        return cls(
            kite_api_key=os.getenv("KITE_API_KEY", "").strip(),
            kite_access_token=os.getenv("KITE_ACCESS_TOKEN", "").strip(),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            tz=os.getenv("TZ", "Asia/Kolkata").strip(),
            use_yfinance_fallback=os.getenv("USE_YFINANCE_FALLBACK", "true").lower() in ("true", "1", "yes"),
            intraday_option_selling=mode,
            trade_lots=lots,
            auto_trade=os.getenv("AUTO_TRADE", "true").lower() in ("true", "1", "yes"),
            product_type=os.getenv("PRODUCT_TYPE", "MIS").strip().upper()
        )

    @property
    def trading_mode(self) -> str:
        return self.intraday_option_selling

    @property
    def is_live_trading(self) -> bool:
        return self.intraday_option_selling == "LIVE" and self.auto_trade

    @property
    def is_paper_trading(self) -> bool:
        return self.intraday_option_selling == "PAPER" and self.auto_trade

    @property
    def has_kite_creds(self) -> bool:
        return bool(self.kite_api_key and self.kite_access_token)

    @property
    def has_telegram_creds(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)
