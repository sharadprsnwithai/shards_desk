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

    @classmethod
    def load(cls) -> "Config":
        return cls(
            kite_api_key=os.getenv("KITE_API_KEY", "").strip(),
            kite_access_token=os.getenv("KITE_ACCESS_TOKEN", "").strip(),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            tz=os.getenv("TZ", "Asia/Kolkata").strip(),
            use_yfinance_fallback=os.getenv("USE_YFINANCE_FALLBACK", "true").lower() in ("true", "1", "yes")
        )

    @property
    def has_kite_creds(self) -> bool:
        return bool(self.kite_api_key and self.kite_access_token)

    @property
    def has_telegram_creds(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)
