import sys
import subprocess
from pathlib import Path

PYTHON_313 = r"C:\Users\sharad\AppData\Local\Programs\Python\Python313\python.exe"

def main():
    script_path = Path(__file__).resolve().parent / "scheduler.py"
    print(f"[Launcher] Starting Kronos Live Scheduler with 2 lots in LIVE mode...")
    print(f"[Launcher] Using Python: {PYTHON_313}")
    subprocess.run([PYTHON_313, str(script_path)], check=True)

if __name__ == "__main__":
    main()
