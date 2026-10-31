import pytest
from unittest.mock import MagicMock, patch
from zerodha_executor import ZerodhaOrderExecutor


def test_executor_init_paper_mode():
    executor = ZerodhaOrderExecutor(mode="PAPER", lots=2)
    assert executor.is_paper is True
    assert executor.is_live is False
    assert executor.lots == 2


def test_executor_init_live_mode():
    executor = ZerodhaOrderExecutor(mode="LIVE", lots=3)
    assert executor.is_live is True
    assert executor.is_paper is False
    assert executor.lots == 3


def test_executor_env_var_intraday_option_selling(monkeypatch):
    from config import Config
    # 1. Test INTRADAY_OPTION_SELLING=LIVE
    monkeypatch.setenv("INTRADAY_OPTION_SELLING", "LIVE")
    cfg = Config.load()
    assert cfg.intraday_option_selling == "LIVE"
    assert cfg.is_live_trading is True
    executor = ZerodhaOrderExecutor()
    assert executor.mode == "LIVE"

    # 2. Test INTRADAY_OPTION_SELLING=PAPER
    monkeypatch.setenv("INTRADAY_OPTION_SELLING", "PAPER")
    cfg2 = Config.load()
    assert cfg2.intraday_option_selling == "PAPER"
    assert cfg2.is_paper_trading is True
    executor2 = ZerodhaOrderExecutor()
    assert executor2.mode == "PAPER"


def test_executor_margin_check():
    executor = ZerodhaOrderExecutor(mode="PAPER")
    assert executor.check_margin_availability(required_lots=5) is True

    # Test LIVE margin check with mock Kite
    executor_live = ZerodhaOrderExecutor(mode="LIVE")
    mock_kite = MagicMock()
    # Sufficient cash: ₹10,00,000 > required 2 * 135,000 = 270,000
    mock_kite.margins.return_value = {"net": 1000000.0}
    executor_live.kite = mock_kite
    assert executor_live.check_margin_availability(required_lots=2) is True

    # Insufficient cash: ₹1,00,000 < required 2 * 135,000 = 270,000
    mock_kite.margins.return_value = {"net": 100000.0}
    assert executor_live.check_margin_availability(required_lots=2) is False


def test_executor_paper_trade_entry_and_exit(tmp_path):
    with patch("zerodha_executor.ORDER_BOOK_FILE", tmp_path / "order_book.json"):
        with patch("zerodha_executor.ACTIVE_ORDERS_FILE", tmp_path / "active_orders.json"):
            with patch("zerodha_executor.TRADE_LEDGER_CSV", tmp_path / "trade_ledger.csv"):
                with patch("zerodha_executor.ZerodhaOrderExecutor._is_market_hours", return_value=True):
                    executor = ZerodhaOrderExecutor(mode="PAPER", lots=1)
                    executor._lot_size = 25  # Lock lot size for test

                    sample_pos = {
                        "date": "2026-10-30",
                        "spot_10am": 22960.0,
                        "legs": [
                            {
                                "action": "SELL",
                                "strike": 23150,
                                "type": "CE",
                                "tradingsymbol": "NIFTY26N0323150CE",
                                "entry_premium": 50.0,
                                "sl_premium": 75.0,
                                "target_premium": 10.0,
                                "status": "ACTIVE"
                            }
                        ]
                    }

                    # 1. Test Entry
                    res = executor.execute_entry(sample_pos)
                    assert res["mode"] == "PAPER"
                    assert len(res["orders"]) == 1
                    assert res["orders"][0]["status"] == "COMPLETE"
                    assert res["orders"][0]["fill_price"] == 50.0

                    # 2. Test SL Exit
                    leg = sample_pos["legs"][0]
                    leg["exit_premium"] = 75.5
                    sl_res = executor.execute_sl_tp_exit(leg, trigger_type="STOP_LOSS")
                    assert sl_res["action"] == "BUY"
                    assert sl_res["fill_price"] == 75.5
                    assert sl_res["status"] == "COMPLETE"

                    # 3. Test EOD Square-off
                    leg["status"] = "ACTIVE"
                    leg["exit_premium"] = 25.0
                    eod_res = executor.execute_eod_squareoff(sample_pos)
                    assert len(eod_res["closed_orders"]) == 1
                    assert eod_res["closed_orders"][0]["trigger_type"] == "3:00 PM EOD"


def test_ledger_records_correct_symbol_per_leg(tmp_path):
    with patch("zerodha_executor.ORDER_BOOK_FILE", tmp_path / "ob.json"), \
         patch("zerodha_executor.ACTIVE_ORDERS_FILE", tmp_path / "ao.json"), \
         patch("zerodha_executor.TRADE_LEDGER_CSV", tmp_path / "ledger.csv"), \
         patch.object(ZerodhaOrderExecutor, "_is_market_hours", return_value=True):
        executor = ZerodhaOrderExecutor(mode="PAPER", lots=1)
        executor._lot_size = 75
        pos = {
            "date": "2025-06-01",
            "legs": [
                {"action": "SELL", "strike": 24400, "type": "CE",
                 "tradingsymbol": "NIFTY25JUN24400CE",
                 "entry_premium": 50.0, "exit_premium": 10.0,
                 "quantity": 75, "status": "CLOSED", "points_gained": 40.0},
                {"action": "SELL", "strike": 24000, "type": "PE",
                 "tradingsymbol": "NIFTY25JUN24000PE",
                 "entry_premium": 45.0, "exit_premium": 9.0,
                 "quantity": 75, "status": "CLOSED", "points_gained": 36.0},
            ]
        }
        executor._record_trade_ledger(pos)

    import pandas as pd
    df = pd.read_csv(tmp_path / "ledger.csv")
    assert len(df) == 2
    assert "NIFTY25JUN24400CE" in df["tradingsymbol"].values
    assert "NIFTY25JUN24000PE" in df["tradingsymbol"].values


def test_eod_squareoff_uses_leg_quantity_not_config_lots(tmp_path):
    with patch("zerodha_executor.ORDER_BOOK_FILE", tmp_path / "ob.json"), \
         patch("zerodha_executor.ACTIVE_ORDERS_FILE", tmp_path / "ao.json"), \
         patch("zerodha_executor.TRADE_LEDGER_CSV", tmp_path / "ledger.csv"), \
         patch.object(ZerodhaOrderExecutor, "_is_market_hours", return_value=True), \
         patch("zerodha_executor.send_telegram_message"):
        # Entry was 1 lot (qty=75), but config says lots=2 now
        executor = ZerodhaOrderExecutor(mode="PAPER", lots=2)
        executor._lot_size = 75
        pos = {
            "date": "2025-06-01",
            "legs": [
                {"action": "SELL", "strike": 24400, "type": "CE",
                 "tradingsymbol": "NIFTY25JUN24400CE",
                 "entry_premium": 50.0, "exit_premium": 20.0,
                 "quantity": 75,
                 "status": "ACTIVE"}
            ]
        }
        result = executor.execute_eod_squareoff(pos)
    assert result["closed_orders"][0]["quantity"] == 75


def test_ledger_skips_leg_with_none_tradingsymbol(tmp_path):
    """A leg with tradingsymbol=None (Kite unavailable) must not corrupt the ledger."""
    with patch("zerodha_executor.ORDER_BOOK_FILE", tmp_path / "ob.json"), \
         patch("zerodha_executor.ACTIVE_ORDERS_FILE", tmp_path / "ao.json"), \
         patch("zerodha_executor.TRADE_LEDGER_CSV", tmp_path / "ledger.csv"), \
         patch.object(ZerodhaOrderExecutor, "_is_market_hours", return_value=True):
        executor = ZerodhaOrderExecutor(mode="PAPER", lots=1)
        executor._lot_size = 75
        pos = {
            "date": "2025-06-01",
            "legs": [
                {"action": "SELL", "strike": 24400, "type": "CE",
                 "tradingsymbol": "NIFTY25JUN24400CE",
                 "entry_premium": 50.0, "exit_premium": 10.0,
                 "quantity": 75, "status": "CLOSED", "points_gained": 40.0},
                {"action": "SELL", "strike": 24000, "type": "PE",
                 "tradingsymbol": None,   # simulates Kite unavailable
                 "entry_premium": 45.0, "exit_premium": 9.0,
                 "quantity": 75, "status": "CLOSED", "points_gained": 36.0},
            ]
        }
        executor._record_trade_ledger(pos)

    import pandas as pd
    df = pd.read_csv(tmp_path / "ledger.csv")
    # Only the leg with a valid symbol should be recorded
    assert len(df) == 1
    assert df.iloc[0]["tradingsymbol"] == "NIFTY25JUN24400CE"
