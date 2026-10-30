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
