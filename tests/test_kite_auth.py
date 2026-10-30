import os
import pyotp
from pathlib import Path
from kite_auth import save_access_token_to_env, auto_login_with_totp

def test_pyotp_totp_generation():
    # Standard RFC 6238 base32 secret
    secret = "JBSWY3DPEHPK3PXP"
    totp = pyotp.TOTP(secret)
    token = totp.now()
    assert len(token) == 6
    assert token.isdigit()

def test_save_access_token_to_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("KITE_API_KEY=test_key\nKITE_ACCESS_TOKEN=old_token\n", encoding="utf-8")
    
    save_access_token_to_env("new_fresh_token", env_file_path=env_file)
    content = env_file.read_text(encoding="utf-8")
    assert "KITE_ACCESS_TOKEN=new_fresh_token" in content
    assert "old_token" not in content

def test_auto_login_skips_when_missing_creds(tmp_path):
    empty_env = tmp_path / ".env"
    empty_env.write_text("", encoding="utf-8")
    res = auto_login_with_totp(user_id="", password="", totp_key="", env_file_path=empty_env)
    assert res is None
