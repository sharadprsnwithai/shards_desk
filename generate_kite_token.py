import os
from pathlib import Path
from dotenv import load_dotenv
from kiteconnect import KiteConnect

env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_file)

api_key = os.getenv("KITE_API_KEY", "").strip()
api_secret = os.getenv("KITE_API_SECRET", "").strip()

print("=" * 70)
print("  🔑  ZERODHA KITE CONNECT — DAILY ACCESS TOKEN GENERATOR")
print("=" * 70)

if not api_key:
    api_key = input("Enter your Kite API Key: ").strip()

if not api_secret:
    api_secret = input("Enter your Kite API Secret: ").strip()

kite = KiteConnect(api_key=api_key)
login_url = kite.login_url()

print(f"\n1. Open this URL in your browser to login:")
print(f"   {login_url}\n")
print("2. After login, you will be redirected to your Redirect URL.")
print("   It will look like: https://127.0.0.1/?action=login&status=success&request_token=XXXXXX\n")

req_input = input("Paste the FULL redirect URL (or just the request_token): ").strip()

# Extract request_token if full URL was pasted
if "request_token=" in req_input:
    import urllib.parse
    parsed = urllib.parse.urlparse(req_input)
    params = urllib.parse.parse_qs(parsed.query)
    request_token = params.get("request_token", [req_input])[0]
else:
    request_token = req_input

try:
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    print(f"\n✅ SUCCESS! Generated Today's Access Token: {access_token[:6]}...{access_token[-4:]}")

    # Update .env automatically
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        new_lines = []
        found_tok = False
        found_secret = False
        for line in lines:
            if line.startswith("KITE_ACCESS_TOKEN="):
                new_lines.append(f"KITE_ACCESS_TOKEN={access_token}")
                found_tok = True
            elif line.startswith("KITE_API_SECRET="):
                new_lines.append(f"KITE_API_SECRET={api_secret}")
                found_secret = True
            else:
                new_lines.append(line)
        if not found_tok:
            new_lines.append(f"KITE_ACCESS_TOKEN={access_token}")
        if not found_secret:
            new_lines.append(f"KITE_API_SECRET={api_secret}")

        env_file.write_text("\n".join(new_lines), encoding="utf-8")
        print("💾 Updated .env with today's fresh KITE_ACCESS_TOKEN!")
    
except Exception as e:
    print(f"\n❌ Error generating session: {e}")
