import time
import logging
from datetime import datetime, date, timedelta
import pytz

from monthly_engine import run_monthly_analysis
from daily_engine import run_intraday_entry, run_intraday_exit, monitor_intraday_sl_tp
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IST_TZ = pytz.timezone("Asia/Kolkata")


def is_last_day_of_month(d: date) -> bool:
    """
    Returns True if the given date is the final calendar day of the month.
    Works reliably for all months and leap years.
    """
    tomorrow = d + timedelta(days=1)
    return tomorrow.day == 1


def check_scheduled_triggers(
    now_ist: datetime,
    state: dict,
    run_intraday_entry_fn=run_intraday_entry,
    run_intraday_exit_fn=run_intraday_exit,
    monitor_intraday_fn=monitor_intraday_sl_tp,
    train_monthly_fn=None,
    run_monthly_fn=run_monthly_analysis,
) -> dict:
    """
    Evaluates current time against scheduled rules and triggers corresponding pipelines:
    1. Weekday 10:00 AM -> 4-OTM Intraday Entry
    2. Weekday 3:00 PM (15:00) -> Intraday Exit & PnL
    3. Last Day of Month 4:00 PM (16:00) -> Retrain Multi-Asset Monthly LSTM
    4. Last Day of Month 5:00 PM (17:00) -> 30-Day Monthly Positional Analysis
    """
    today = now_ist.date()
    hour = now_ist.hour
    minute = now_ist.minute
    weekday = now_ist.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun

    # 1. 10:00 AM IST Weekday Intraday Entry (Mon-Fri)
    if weekday < 5 and hour == 10 and minute == 0 and state.get("last_entry_date") != today:
        logger.info("[Scheduler] Triggering 10:00 AM 4-OTM Entry Signal & Telegram Alert...")
        if run_intraday_entry_fn:
            run_intraday_entry_fn(send_telegram=True)
        state["last_entry_date"] = today

    # 1b. Real-Time Monitor: Weekday 10:01 AM -> 2:59 PM (Mon-Fri) checks 50% SL & 80% Target
    is_active_session = (weekday < 5) and (
        (hour == 10 and minute > 0) or
        (11 <= hour < 15)
    )
    if is_active_session and monitor_intraday_fn:
        try:
            monitor_intraday_fn(send_telegram=True)
        except Exception as e:
            logger.warning(f"[Scheduler] Intraday SL/TP monitor error: {e}")

    # 2. 3:00 PM (15:00) IST Weekday Intraday Exit & PnL (Mon-Fri)
    if weekday < 5 and hour == 15 and minute == 0 and state.get("last_exit_date") != today:
        logger.info("[Scheduler] Triggering 3:00 PM Option Exit & PnL Alert...")
        if run_intraday_exit_fn:
            run_intraday_exit_fn(send_telegram=True)
        state["last_exit_date"] = today

    # 3. 4:00 PM (16:00) IST Last Day of Month: Auto-Retrain Monthly Multi-Asset LSTM
    if is_last_day_of_month(today) and hour == 16 and minute == 0 and state.get("last_monthly_train_date") != today:
        logger.info("[Scheduler] Triggering Last-Day-of-Month 4:00 PM Multi-Asset Monthly LSTM Retraining...")
        if train_monthly_fn:
            try:
                train_monthly_fn(epochs=35)
                logger.info("[Scheduler] Monthly LSTM weights updated successfully.")
            except Exception as e:
                logger.error(f"[Scheduler] Monthly LSTM retraining failed: {e}", exc_info=True)
        else:
            try:
                from monthly_direction_engine.train_monthly_lstm import train_multi_asset_monthly_lstm
                train_multi_asset_monthly_lstm(epochs=35)
                logger.info("[Scheduler] Monthly LSTM weights updated successfully.")
            except Exception as e:
                logger.error(f"[Scheduler] Monthly LSTM retraining failed: {e}", exc_info=True)
        state["last_monthly_train_date"] = today

    # 4. 5:00 PM (17:00) IST Last Day of Month Positional Run
    if is_last_day_of_month(today) and hour == 17 and minute == 0 and state.get("last_monthly_run_date") != today:
        logger.info("[Scheduler] Triggering Last-Day-of-Month 5:00 PM Monthly Option Desk & Telegram Alert...")
        if run_monthly_fn:
            run_monthly_fn(send_telegram=True)
        state["last_monthly_run_date"] = today

    return state


def start_scheduler():
    """
    Continuous background scheduler:
    1. Every weekday (Mon-Fri) at 10:00 AM IST:
       - Runs Hybrid Engine (Kronos + LSTM + VWAP)
       - Selects 4-OTM Tuesday weekly expiry options
       - Dispatches Telegram entry signal with exact strikes & entry LTP
    2. Every weekday (Mon-Fri) at 3:00 PM (15:00) IST:
       - Checks live option quotes
       - Calculates exact Option Premium PnL (points gained/lost)
       - Dispatches Telegram exit alert and squares off
    3. On the last day of every month at 4:00 PM (16:00) IST:
       - Automatically retrains Monthly Multi-Asset LSTM on fresh daily data
    4. On the last day of every month at 5:00 PM (17:00) IST:
       - Executes 30-Day Monthly Positional Option Desk & Telegram Alert
    """
    cfg = Config.load()
    logger.info("=" * 80)
    logger.info("  🚀  KRONOS TRADING DESK SCHEDULER STARTED")
    logger.info("  Timezone   : Asia/Kolkata (IST)")
    logger.info(f"  Execution  : [INTRADAY_OPTION_SELLING={cfg.intraday_option_selling}] | Auto-Trade: {cfg.auto_trade} | Lots: {cfg.trade_lots}")
    logger.info("  Schedule 1 : Mon-Fri @ 10:00 AM IST -> 4-OTM Entry & Order Placement")
    logger.info("  Monitor    : Mon-Fri @ 10:01 AM - 2:59 PM -> Real-Time 50% SL / 80% TP Watch")
    logger.info("  Schedule 2 : Mon-Fri @ 3:00 PM IST  -> EOD Square-Off & Final PnL Report")
    logger.info("  Schedule 3 : Last Day of Month @ 4:00 PM IST -> Auto-Retrain Monthly LSTM")
    logger.info("  Schedule 4 : Last Day of Month @ 5:00 PM IST -> Monthly Option Desk Alert")
    logger.info("=" * 80)

    state = {
        "last_entry_date": None,
        "last_exit_date": None,
        "last_monthly_train_date": None,
        "last_monthly_run_date": None,
    }

    while True:
        try:
            now_ist = datetime.now(IST_TZ)
            check_scheduled_triggers(now_ist, state)
        except Exception as e:
            logger.error(f"[Scheduler] Loop encountered error: {e}", exc_info=True)

        # Sleep 20 seconds between checks
        time.sleep(20)


if __name__ == "__main__":
    start_scheduler()
