"""
Zerodha Automated Order Execution Service (Intraday 4-OTM Desk)
================================================================
Executes 10:00 AM 4-OTM NIFTY option entries, real-time 50% SL / 80% TP exits,
and 3:00 PM EOD square-offs via Zerodha Kite Connect API.

Modes:
  - PAPER : Simulates live order fills, records order book & trade ledger (Safe / Default).
  - LIVE  : Real order placement on Kite Connect (NFO MIS) with margin verification.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, time as dtime
import pytz
import pandas as pd

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR))

from config import Config
from data_provider import DataProvider
from telegram_notifier import send_telegram_message

logger = logging.getLogger(__name__)

ORDER_BOOK_FILE = CUR_DIR / "data_cache" / "order_book.json"
TRADE_LEDGER_CSV = CUR_DIR / "data_cache" / "trade_ledger.csv"
ACTIVE_ORDERS_FILE = CUR_DIR / "data_cache" / "active_orders.json"


class ZerodhaOrderExecutor:
    """
    Automated Order Execution Service for Intraday Option Selling.
    Handles PAPER and LIVE trade lifecycles:
      1. 10:00 AM Entry   -> Places SELL orders (Market MIS) for 4-OTM strikes.
      2. Mid-Day SL/TP    -> Places BUY orders (Market MIS) to square off triggered legs.
      3. 3:00 PM EOD      -> Places BUY orders (Market MIS) to square off remaining legs.
    """

    def __init__(self, mode: Optional[str] = None, lots: Optional[int] = None):
        self.cfg = Config.load()
        self.mode = (mode or self.cfg.intraday_option_selling or self.cfg.trading_mode).upper()
        if self.mode not in ("LIVE", "PAPER"):
            self.mode = "PAPER"

        self.lots = lots if lots is not None else self.cfg.trade_lots
        self.product = self.cfg.product_type or "MIS"
        self.dp = DataProvider()
        self.kite = self.dp.kite

        # Cached NIFTY lot size
        self._lot_size: Optional[int] = None

        logger.info(f"[Executor] Initialized in {self.mode} mode | Target Lots: {self.lots} | Product: {self.product}")

    @property
    def is_live(self) -> bool:
        return self.mode == "LIVE"

    @property
    def is_paper(self) -> bool:
        return self.mode == "PAPER"

    def get_nifty_lot_size(self) -> int:
        """
        Dynamically fetches current NIFTY lot size from Kite Connect instruments.
        Falls back to 65 if Kite is not connected or instruments fetch fails.
        """
        if self._lot_size is not None:
            return self._lot_size

        if self.kite:
            try:
                instruments = self.kite.instruments("NFO")
                for inst in instruments:
                    if inst.get("name") == "NIFTY" and inst.get("instrument_type") in ("CE", "PE"):
                        self._lot_size = int(inst["lot_size"])
                        logger.info(f"[Executor] Fetched dynamic NIFTY lot size from Kite: {self._lot_size}")
                        return self._lot_size
            except Exception as e:
                logger.warning(f"[Executor] Could not fetch instruments from Kite: {e}")

        # Fallback to standard 65
        self._lot_size = 65
        return self._lot_size

    def check_margin_availability(self, required_lots: int) -> bool:
        """
        In LIVE mode: Verifies if Zerodha account has sufficient free cash margin.
        Assumes ~₹55,000 per lot for hedged credit spreads / Iron Condor.
        In PAPER mode: Always returns True.
        """
        if self.is_paper:
            return True

        if not self.kite:
            logger.error("[Executor] Cannot verify margin: Kite Connect not authenticated.")
            return False

        try:
            margins = self.kite.margins("equity")
            available_cash = float(margins.get("net", 0.0))
            # Hedged credit spread requirement (~₹55,000 per lot)
            required_margin = required_lots * 55_000.0

            logger.info(
                f"[Executor] Margin Check: Available=₹{available_cash:,.2f} | "
                f"Required (~{required_lots} lots)=₹{required_margin:,.2f}"
            )

            if available_cash < required_margin:
                logger.error(
                    f"[Executor] ❌ INSUFFICIENT MARGIN! Available: ₹{available_cash:,.2f} < "
                    f"Required: ₹{required_margin:,.2f}. Aborting live order execution."
                )
                return False

            return True
        except Exception as e:
            logger.error(f"[Executor] Error fetching Kite margins: {e}")
            return False

    # ── Trading hours guard ────────────────────────────────────────────────────

    @staticmethod
    def _is_market_hours() -> bool:
        """
        Returns True only when IST time is within NSE trading hours
        (9:15 AM – 3:30 PM, Mon–Fri).  All automated order placement is
        blocked outside this window to prevent accidental fills on restart.
        """
        IST = pytz.timezone("Asia/Kolkata")
        now = datetime.now(IST)
        if now.weekday() >= 5:          # Saturday / Sunday
            return False
        t = now.time()
        return dtime(9, 15) <= t <= dtime(15, 30)

    def execute_entry(self, position_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes 10:00 AM Entry orders for all legs in the position payload.
        Places SELL Market MIS orders (LIVE) or records simulated fills (PAPER).
        """
        if not self._is_market_hours():
            msg = "[Executor] ⛔ Order execution BLOCKED: outside NSE market hours (9:15 AM – 3:30 PM IST, Mon–Fri)."
            logger.warning(msg)
            send_telegram_message(f"⛔ <b>Order Blocked (Outside Market Hours):</b>\n{msg}")
            return {"status": "OUTSIDE_MARKET_HOURS", "orders": []}

        legs = position_payload.get("legs", [])
        if not legs:
            logger.warning("[Executor] No legs provided in position payload.")
            return {"status": "EMPTY", "orders": []}

        lot_size = self.get_nifty_lot_size()
        total_qty = self.lots * lot_size

        print("\n" + "=" * 90)
        mode_tag = "🔴 LIVE ZERODHA EXECUTION" if self.is_live else "📝 PAPER TRADE EXECUTION"
        print(f"  ⚡  {mode_tag} — 10:00 AM ENTRY")
        print(f"  Mode: {self.mode} | Lots: {self.lots} (Total Qty: {total_qty}) | Product: {self.product}")
        print("=" * 90 + "\n")

        # Pre-execution safety check for LIVE mode
        if self.is_live:
            if not self.kite:
                err_msg = "[Executor] ❌ Cannot trade LIVE: Kite credentials missing or expired. Run generate_kite_token.py."
                logger.error(err_msg)
                send_telegram_message(f"⚠️ <b>LIVE ORDER REJECTED:</b>\n{err_msg}")
                return {"status": "AUTH_FAILED", "orders": []}

            if not self.check_margin_availability(self.lots):
                send_telegram_message(
                    f"⚠️ <b>LIVE ORDER REJECTED (INSUFFICIENT MARGIN):</b>\n"
                    f"Required: ~₹{self.lots * 55000:,.0f} for {self.lots} lot(s). Check Zerodha funds."
                )
                return {"status": "MARGIN_FAILED", "orders": []}

        executed_orders = []

        for leg in legs:
            tsym = leg.get("tradingsymbol", "")
            action = leg.get("action", "SELL")
            leg_tx_type = self.kite.TRANSACTION_TYPE_BUY if action == "BUY" else self.kite.TRANSACTION_TYPE_SELL
            opt_type = leg.get("type", "CE")
            strike = leg.get("strike", 0)
            entry_prem = float(leg.get("entry_premium", 0.0))

            if not tsym:
                logger.warning(f"[Executor] Missing tradingsymbol for strike {strike} {opt_type}. Skipping.")
                continue

            order_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": self.mode,
                "action": action,
                "tradingsymbol": tsym,
                "strike": strike,
                "type": opt_type,
                "quantity": total_qty,
                "lots": self.lots,
                "product": self.product,
                "order_id": None,
                "fill_price": entry_prem,
                "status": "PENDING"
            }

            if self.is_live:
                try:
                    logger.info(f"[Executor] Placing LIVE order on Kite: SELL {total_qty}x {tsym} (MIS)...")
                    order_id = self.kite.place_order(
                        variety=self.kite.VARIETY_REGULAR,
                        exchange=self.kite.EXCHANGE_NFO,
                        tradingsymbol=tsym,
                        transaction_type=leg_tx_type,
                        quantity=total_qty,
                        order_type=self.kite.ORDER_TYPE_MARKET,
                        product=self.product,
                        tag="INT4OTM"
                    )
                    order_record["order_id"] = str(order_id)
                    order_record["status"] = "COMPLETE"

                    # Fetch actual executed average price from order history
                    try:
                        time.sleep(0.5)
                        history = self.kite.order_history(order_id)
                        if history:
                            latest = history[-1]
                            avg_p = float(latest.get("average_price", 0.0))
                            if avg_p > 0:
                                order_record["fill_price"] = avg_p
                                leg["entry_premium"] = avg_p
                                leg["sl_premium"] = round(avg_p * 1.50, 1)
                                leg["target_premium"] = round(avg_p * 0.20, 1)
                    except Exception as he:
                        logger.warning(f"[Executor] Could not fetch order history for {order_id}: {he}")

                    logger.info(f"[Executor] ✅ LIVE Order Placed: ID={order_id} @ ₹{order_record['fill_price']:.2f}")

                except Exception as e:
                    order_record["status"] = "REJECTED"
                    order_record["error"] = str(e)
                    logger.error(f"[Executor] ❌ LIVE Order Failed for {tsym}: {e}")
                    send_telegram_message(f"⚠️ <b>Order Placement Error ({tsym}):</b>\n<code>{e}</code>")
                    leg["status"] = "ENTRY_REJECTED"

            else:  # PAPER MODE
                order_record["order_id"] = f"PAPER_{int(time.time() * 1000)}"
                order_record["status"] = "COMPLETE"
                order_record["fill_price"] = entry_prem
                logger.info(f"[Executor] 📝 PAPER Order Recorded: SELL {total_qty}x {tsym} @ ₹{entry_prem:.2f}")

            # Attach execution details to leg object
            leg["order_id"] = order_record["order_id"]
            leg["fill_price"] = order_record["fill_price"]
            leg["quantity"] = total_qty
            leg["execution_mode"] = self.mode

            executed_orders.append(order_record)
            self._append_to_order_book(order_record)

        # Alert when a sold leg filled but its hedge wing was rejected (unhedged exposure)
        filled_sold = [l for l in legs if l.get("action") == "SELL" and l.get("status") == "ACTIVE"]
        rejected_wing = [l for l in legs if l.get("action") == "BUY" and l.get("status") == "ENTRY_REJECTED"]
        if filled_sold and rejected_wing:
            sym = filled_sold[0].get("tradingsymbol", "?")
            send_telegram_message(
                f"⚠️ <b>UNHEDGED ALERT — {sym}</b>\n"
                f"Sold leg filled; hedge wing rejected. Short position is unhedged — do not hold overnight."
            )

        # Dispatch execution confirmation to Telegram
        self._send_entry_telegram_notice(executed_orders)

        summary = {
            "mode": self.mode,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lots": self.lots,
            "quantity": total_qty,
            "orders": executed_orders
        }
        self._save_active_orders(summary)
        return summary

    def execute_sl_tp_exit(self, leg: Dict[str, Any], trigger_type: str) -> Dict[str, Any]:
        """
        Executes immediate market BUY order to square off a single leg when
        its 50% Stop-Loss or 80% Profit Target is triggered during market hours.
        """
        if not self._is_market_hours():
            logger.warning(
                f"[Executor] ⛔ SL/TP exit BLOCKED for {leg.get('tradingsymbol', '?')}: "
                "outside NSE market hours."
            )
            return {"status": "OUTSIDE_MARKET_HOURS"}

        # Long hedge wings have no SL/TP to monitor — nothing to square off
        if leg.get("action") == "BUY":
            return {"status": "SKIPPED_LONG_LEG"}

        tsym = leg.get("tradingsymbol", "")
        lot_size = self.get_nifty_lot_size()
        total_qty = leg.get("quantity", self.lots * lot_size)
        trigger_price = float(leg.get("exit_premium", 0.0))

        order_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.mode,
            "action": "BUY",
            "trigger_type": trigger_type,
            "tradingsymbol": tsym,
            "strike": leg.get("strike", 0),
            "type": leg.get("type", "CE"),
            "quantity": total_qty,
            "product": self.product,
            "order_id": None,
            "fill_price": trigger_price,
            "status": "PENDING"
        }

        mode_label = "🔴 LIVE" if self.is_live else "📝 PAPER"
        logger.info(f"[Executor] {mode_label} Square-off Triggered ({trigger_type}): BUY {total_qty}x {tsym}...")

        if self.is_live:
            if not self.kite:
                logger.error("[Executor] Cannot square-off LIVE: Kite client unavailable.")
                return order_record

            try:
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NFO,
                    tradingsymbol=tsym,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=total_qty,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    product=self.product,
                    tag="INT4OTM_SLTP"
                )
                order_record["order_id"] = str(order_id)
                order_record["status"] = "COMPLETE"

                # Check actual fill price
                try:
                    time.sleep(0.5)
                    history = self.kite.order_history(order_id)
                    if history:
                        avg_p = float(history[-1].get("average_price", trigger_price))
                        if avg_p > 0:
                            order_record["fill_price"] = avg_p
                            leg["exit_premium"] = avg_p
                except Exception:
                    pass

                logger.info(f"[Executor] ✅ LIVE Exit Complete: ID={order_id} @ ₹{order_record['fill_price']:.2f}")

            except Exception as e:
                order_record["status"] = "REJECTED"
                order_record["error"] = str(e)
                logger.error(f"[Executor] ❌ LIVE Square-off Failed for {tsym}: {e}")
                send_telegram_message(f"⚠️ <b>SL/TP Square-off Error ({tsym}):</b>\n<code>{e}</code>")

        else:  # PAPER MODE
            order_record["order_id"] = f"PAPER_EXIT_{int(time.time() * 1000)}"
            order_record["status"] = "COMPLETE"
            order_record["fill_price"] = trigger_price

        self._append_to_order_book(order_record)
        return order_record

    def execute_eod_squareoff(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes 3:00 PM square-off BUY orders for any legs that are still ACTIVE.
        """
        if not self._is_market_hours():
            logger.warning(
                "[Executor] ⛔ EOD square-off BLOCKED: outside NSE market hours."
            )
            return {"status": "OUTSIDE_MARKET_HOURS", "closed_orders": []}

        legs = position.get("legs", [])
        lot_size = self.get_nifty_lot_size()

        closed_orders = []

        for leg in legs:
            # Skip if already squared off via SL/TP, already CLOSED, or whose
            # entry was rejected (nothing was ever bought/sold — no order to close).
            if leg.get("status") in ("STOP_LOSS_HIT", "TARGET_HIT", "CLOSED", "ENTRY_REJECTED"):
                continue

            tsym = leg.get("tradingsymbol", "")
            if not tsym:
                continue

            qty = leg.get("quantity")
            if qty is None:
                logger.error(f"[Executor] EOD squareoff: missing quantity on leg {tsym}. Skipping.")
                continue

            # Close long wings with SELL; cover short legs with BUY
            if leg.get("action") == "BUY":
                tx_type = self.kite.TRANSACTION_TYPE_SELL
                close_action = "SELL"
                pts = float(leg.get("exit_premium", 0.0) or 0.0) - float(leg.get("entry_premium", 0.0) or 0.0)
            else:
                tx_type = self.kite.TRANSACTION_TYPE_BUY
                close_action = "BUY"
                pts = float(leg.get("entry_premium", 0.0) or 0.0) - float(leg.get("exit_premium", 0.0) or 0.0)

            order_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": self.mode,
                "action": close_action,
                "trigger_type": "3:00 PM EOD",
                "tradingsymbol": tsym,
                "quantity": qty,
                "product": self.product,
                "order_id": None,
                "fill_price": float(leg.get("exit_premium", 0.0)),
                "status": "PENDING"
            }

            if self.is_live:
                if self.kite:
                    try:
                        order_id = self.kite.place_order(
                            variety=self.kite.VARIETY_REGULAR,
                            exchange=self.kite.EXCHANGE_NFO,
                            tradingsymbol=tsym,
                            transaction_type=tx_type,
                            quantity=order_record["quantity"],
                            order_type=self.kite.ORDER_TYPE_MARKET,
                            product=self.product,
                            tag="INT4OTM_EOD"
                        )
                        order_record["order_id"] = str(order_id)
                        order_record["status"] = "COMPLETE"
                        logger.info(f"[Executor] ✅ LIVE EOD Square-off: ID={order_id} ({tsym})")
                    except Exception as e:
                        order_record["status"] = "REJECTED"
                        order_record["error"] = str(e)
                        logger.error(f"[Executor] ❌ LIVE EOD Square-off Failed: {e}")
            else:
                order_record["order_id"] = f"PAPER_EOD_{int(time.time() * 1000)}"
                order_record["status"] = "COMPLETE"
                order_record["points_gained"] = pts
                leg["points_gained"] = pts

            # Only a complete order closes the leg
            leg["status"] = "CLOSED" if order_record["status"] == "COMPLETE" else "OPEN"
            leg["points_gained"] = pts
            order_record["points_gained"] = pts
            if order_record["status"] == "COMPLETE":
                leg["points_gained"] = pts
                order_record["points_gained"] = pts
            closed_orders.append(order_record)
            self._append_to_order_book(order_record)

        self._record_trade_ledger(position)
        self._clear_active_orders()
        return {"mode": self.mode, "closed_orders": closed_orders}

    # ── Internal Storage & Notifications ──────────────────────────────────────

    def _append_to_order_book(self, order: Dict[str, Any]):
        ORDER_BOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
        orders = []
        if ORDER_BOOK_FILE.exists():
            try:
                orders = json.loads(ORDER_BOOK_FILE.read_text(encoding="utf-8"))
            except Exception:
                orders = []
        orders.append(order)
        ORDER_BOOK_FILE.write_text(json.dumps(orders, indent=2), encoding="utf-8")

    def _save_active_orders(self, summary: Dict[str, Any]):
        ACTIVE_ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_ORDERS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _clear_active_orders(self):
        if ACTIVE_ORDERS_FILE.exists():
            try:
                ACTIVE_ORDERS_FILE.unlink()
            except Exception:
                pass

    def _record_trade_ledger(self, pos: Dict[str, Any]):
        TRADE_LEDGER_CSV.parent.mkdir(parents=True, exist_ok=True)
        date_str = pos.get("date", str(datetime.now().date()))
        lot_size = self.get_nifty_lot_size()

        rows = []
        for leg in pos.get("legs", []):
            tsym = leg.get("tradingsymbol") or ""
            if not tsym:
                logger.warning("[Executor] Skipping ledger entry for leg with missing tradingsymbol.")
                continue
            entry_p = float(leg.get("entry_premium", 0.0))
            exit_p = float(leg.get("exit_premium", 0.0))
            if exit_p <= 0:
                # No exit recorded: trade stays flat — never fabricate PnL.
                pts = 0.0
            elif leg.get("action") == "BUY":
                # Long hedge wing: profit when sold back above entry
                pts = exit_p - entry_p
            else:
                # Short leg: profit when bought back below entry
                pts = entry_p - exit_p
            qty = leg.get("quantity", self.lots * lot_size)
            pnl_rs = pts * qty

            rows.append({
                "date": date_str,
                "mode": self.mode,
                "tradingsymbol": tsym,
                "action": leg.get("action", "SELL"),
                "strike": leg.get("strike", 0),
                "type": leg.get("type", "CE"),
                "quantity": qty,
                "lots": self.lots,
                "entry_premium": entry_p,
                "exit_premium": exit_p,
                "points_gained": pts,
                "pnl_rs": round(pnl_rs, 2),
                "status": leg.get("status", "CLOSED"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        if rows:
            new_df = pd.DataFrame(rows)
            if TRADE_LEDGER_CSV.exists():
                new_df.to_csv(TRADE_LEDGER_CSV, mode="a", header=False, index=False)
            else:
                new_df.to_csv(TRADE_LEDGER_CSV, index=False)
            logger.info(f"[Executor] Appended {len(rows)} trades to {TRADE_LEDGER_CSV.name}")

    def _send_entry_telegram_notice(self, orders: List[Dict[str, Any]]):
        if not orders:
            return

        icon = "🔴" if self.is_live else "📝"
        title = "LIVE ORDER EXECUTED (ZERODHA)" if self.is_live else "PAPER TRADE RECORDED"

        lines = [
            f"{icon} <b>[{title}]</b>",
            f"• Mode: <b>{self.mode}</b> | Lots: <b>{self.lots}</b> (MIS)",
            ""
        ]

        for ord_info in orders:
            tsym = ord_info.get("tradingsymbol", "")
            qty = ord_info.get("quantity", 0)
            fill_p = ord_info.get("fill_price", 0.0)
            status = ord_info.get("status", "")
            oid = ord_info.get("order_id", "")
            oid_str = f" [ID: <code>{oid}</code>]" if oid and not oid.startswith("PAPER") else ""
            lines.append(f"   • SELL <b>{qty}x {tsym}</b> @ <b>₹{fill_p:.2f}</b> ({status}){oid_str}")

        send_telegram_message("\n".join(lines))


def get_executor() -> ZerodhaOrderExecutor:
    """Convenience factory function."""
    return ZerodhaOrderExecutor()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zerodha Automated Execution Service")
    parser.add_argument("--mode", choices=["PAPER", "LIVE"], help="Override execution mode")
    parser.add_argument("--lots", type=int, help="Override trade lot size")
    parser.add_argument("--status", action="store_true", help="Print active position & order book status")
    args = parser.parse_args()

    exec_service = ZerodhaOrderExecutor(mode=args.mode, lots=args.lots)

    if args.status:
        print("\n--- ACTIVE POSITION ---")
        if (CUR_DIR / "data_cache" / "active_intraday_position.json").exists():
            print((CUR_DIR / "data_cache" / "active_intraday_position.json").read_text())
        else:
            print("No active position.")

        print("\n--- ACTIVE ORDERS ---")
        if ACTIVE_ORDERS_FILE.exists():
            print(ACTIVE_ORDERS_FILE.read_text())
        else:
            print("No active orders.")
    else:
        print(f"\n[OK] ZerodhaOrderExecutor initialized successfully in {exec_service.mode} mode.")
        print(f"     NIFTY Dynamic Lot Size : {exec_service.get_nifty_lot_size()}")
        print(f"     Lots Configured        : {exec_service.lots}")
        print(f"     Is Live Trading        : {exec_service.is_live}")
        print(f"     Is Paper Trading       : {exec_service.is_paper}\n")
