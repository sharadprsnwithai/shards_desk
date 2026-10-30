import os
import re
import logging
import urllib.parse
from pathlib import Path
from typing import Optional
import requests
import pyotp
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parent / ".env"


def save_access_token_to_env(access_token: str, env_file_path: Optional[Path] = None) -> bool:
    """Safely updates or appends KITE_ACCESS_TOKEN in the .env file and active process environment."""
    os.environ["KITE_ACCESS_TOKEN"] = access_token
    target_path = env_file_path or ENV_PATH
    if not target_path.exists():
        target_path.write_text(f"KITE_ACCESS_TOKEN={access_token}\n", encoding="utf-8")
        return True

    content = target_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    updated = False
    new_lines = []

    for line in lines:
        if line.strip().startswith("KITE_ACCESS_TOKEN="):
            new_lines.append(f"KITE_ACCESS_TOKEN={access_token}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"KITE_ACCESS_TOKEN={access_token}")

    target_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


def auto_login_with_totp(
    user_id: Optional[str] = None,
    password: Optional[str] = None,
    totp_key: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    env_file_path: Optional[Path] = None
) -> Optional[str]:
    """
    Performs fully automated, headless Zerodha Kite login using TOTP:
    1. Submits User ID + Password to Kite Login API.
    2. Generates 6-digit TOTP code and submits to Two-Factor Auth API.
    3. Traverses OAuth hops (/connect/login -> /connect/finish) to capture request_token.
    4. Exchanges request_token with KiteConnect for fresh access_token.
    5. Saves new access_token to .env file automatically.

    Returns the new access_token string if successful, or None on failure.
    """
    target_env = env_file_path or ENV_PATH
    if target_env.exists():
        load_dotenv(dotenv_path=target_env, override=False)

    uid = user_id if user_id is not None else os.getenv("KITE_USER_ID", "").strip()
    pwd = password if password is not None else os.getenv("KITE_PASSWORD", "").strip()
    tkey = totp_key if totp_key is not None else os.getenv("KITE_TOTP_KEY", "").strip()
    akey = api_key if api_key is not None else os.getenv("KITE_API_KEY", "").strip()
    asecret = api_secret if api_secret is not None else os.getenv("KITE_API_SECRET", "").strip()

    if not all([uid, pwd, tkey, akey, asecret]):
        logger.warning(
            "[KiteAuth] Missing TOTP credentials. Ensure KITE_USER_ID, KITE_PASSWORD, "
            "KITE_TOTP_KEY, KITE_API_KEY, and KITE_API_SECRET are set in .env."
        )
        return None

    # Clean totp key (remove spaces if any)
    tkey = tkey.replace(" ", "").strip()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "X-Kite-Version": "3"
    })

    try:
        # Step 1: Login with User ID and Password
        logger.info("[KiteAuth] Submitting login credentials to Zerodha Kite...")
        login_resp = session.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": uid, "password": pwd},
            timeout=15
        )
        login_json = login_resp.json()
        if login_json.get("status") != "success":
            logger.error(f"[KiteAuth] Login step 1 failed: {login_json.get('message', 'Unknown error')}")
            return None

        request_id = login_json["data"]["request_id"]

        # Step 2: Generate TOTP code and submit TwoFA
        logger.info("[KiteAuth] Generating live TOTP code via pyotp...")
        totp = pyotp.TOTP(tkey)
        totp_code = totp.now()

        twofa_resp = session.post(
            "https://kite.zerodha.com/api/twofa",
            data={
                "user_id": uid,
                "request_id": request_id,
                "twofa_value": totp_code,
                "twofa_type": "totp"
            },
            timeout=15
        )
        twofa_json = twofa_resp.json()
        if twofa_json.get("status") != "success":
            logger.error(f"[KiteAuth] TOTP 2FA verification failed: {twofa_json.get('message', 'Invalid TOTP')}")
            return None

        logger.info("[KiteAuth] 2FA TOTP authentication successful!")

        # Step 3: Trigger Kite Connect OAuth flow (handling hops cleanly without local port connection)
        oauth_url = f"https://kite.zerodha.com/connect/login?api_key={akey}&v=3"
        r_hop1 = session.get(oauth_url, allow_redirects=False, timeout=15)
        loc1 = r_hop1.headers.get("Location") or r_hop1.headers.get("location", "")

        request_token = None

        if "request_token=" in loc1:
            match = re.search(r"request_token=([a-zA-Z0-9]+)", loc1)
            if match:
                request_token = match.group(1)
        elif loc1:
            # Second hop to connect/finish
            if loc1.startswith("/"):
                loc1 = "https://kite.zerodha.com" + loc1
            r_hop2 = session.get(loc1, allow_redirects=False, timeout=15)
            loc2 = r_hop2.headers.get("Location") or r_hop2.headers.get("location", "")
            match = re.search(r"request_token=([a-zA-Z0-9]+)", loc2)
            if match:
                request_token = match.group(1)

        # Fallback if redirected via full URL
        if not request_token:
            try:
                r_full = session.get(oauth_url, allow_redirects=True, timeout=15)
                match = re.search(r"request_token=([a-zA-Z0-9]+)", r_full.url)
                if match:
                    request_token = match.group(1)
            except requests.exceptions.ConnectionError as ce:
                match = re.search(r"request_token=([a-zA-Z0-9]+)", str(ce))
                if match:
                    request_token = match.group(1)

        if not request_token:
            logger.error(f"[KiteAuth] Could not extract request_token from OAuth hops.")
            return None

        logger.info("[KiteAuth] Captured request_token successfully!")

        # Step 4: Exchange request_token for access_token using KiteConnect
        from kiteconnect import KiteConnect
        kc = KiteConnect(api_key=akey)
        data = kc.generate_session(request_token, api_secret=asecret)
        access_token = data.get("access_token")

        if access_token:
            save_access_token_to_env(access_token)
            logger.info("[KiteAuth] [OK] Fresh access token generated and saved to .env!")
            return access_token

    except Exception as e:
        logger.error(f"[KiteAuth] Automated TOTP login error: {e}", exc_info=True)
        return None

    return None
