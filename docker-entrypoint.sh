#!/bin/bash
set -e

# Export cron environment so variables are accessible inside cron execution
printenv | grep -E '^(KITE_|TELEGRAM_|TZ|USE_)' > /etc/environment || true

case "$1" in
  monthly)
    echo "[Docker] Executing 30-Day Monthly Positional Analysis..."
    python /app/monthly_engine.py
    ;;
  intraday)
    echo "[Docker] Executing 10:00 AM Intraday Momentum Analysis..."
    python /app/daily_engine.py
    ;;
  scheduler)
    echo "[Docker] Starting Kronos Background Scheduler (Timezone: $TZ)..."
    echo "  - Schedule 1: Every Weekday (Mon-Fri) at 10:00 AM IST -> 10:00 AM Intraday Momentum Desk"
    echo "  - Schedule 2: Last Day of Month at 5:00 PM (17:00) IST -> 30-Day Monthly Option Selling Desk"
    python /app/scheduler.py
    ;;
  *)
    exec "$@"
    ;;
esac
