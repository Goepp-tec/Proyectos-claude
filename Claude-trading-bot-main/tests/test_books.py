"""Independent paper books (main / observe / baseline) sharing one database."""

from unittest.mock import Mock

import pytest

import config
import database as db
from portfolio_manager import PortfolioManager
from tests.test_portfolio_risk import _buy, _client, _strategy


@pytest.fixture(autouse=True)
def _capital(temp_db, monkeypatch):
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 10_000.0)


def test_books_do_not_see_each_other():
    main = PortfolioManager(_client(), [_strategy("S")])
    base = PortfolioManager(Mock(), [_strategy("S")], book="baseline", simulate_fills=True)
    price, candle = 50_000.0, "2026-09-28 00:00:00+00:00"

    assert base.process_signal(base.strategies["S"], _buy(price), price, 0.6, candle_ts=candle)
    # Same strategy name + same candle in another book is still allowed.
    assert main.process_signal(main.strategies["S"], _buy(price), price, 0.6, candle_ts=candle)

    assert len(db.get_open_positions("S")) == 1                     # default book = main
    assert len(db.get_open_positions("S", book="baseline")) == 1
    assert len(db.get_open_positions("S", book=None)) == 2          # all books

    # Closing on a price move only touches each manager's own book.
    base.check_open_positions(price * 3)                           # hits TP (2x)
    assert len(db.get_open_positions("S", book="baseline")) == 0
    assert len(db.get_open_positions("S")) == 1
    assert db.get_trade_stats("S")["total_trades"] == 0
    assert db.get_trade_stats("S", book="baseline")["total_trades"] == 1


def test_simulated_fills_never_call_the_exchange_client():
    client = Mock()
    pm = PortfolioManager(client, [_strategy("S")], book="observe", simulate_fills=True)
    price = 50_000.0
    assert pm.process_signal(pm.strategies["S"], _buy(price), price, 0.6)
    pm.check_open_positions(price * 3)
    client.place_market_order.assert_not_called()
    pos_fill = db.get_trades("S", book="observe")[0]["entry_price"]
    assert pos_fill == pytest.approx(price * (1 + config.SLIPPAGE))


def test_balance_history_is_per_book():
    db.record_balance(10_000, 0, 0, {}, book="main")
    db.record_balance(12_345, 0, 0, {}, book="baseline")
    assert db.get_latest_balance()["total_balance"] == 10_000
    assert db.get_latest_balance(book="baseline")["total_balance"] == 12_345


def test_non_main_books_do_not_overwrite_strategy_capital_row():
    db.upsert_strategy("S", capital=10_000, params={}, is_active=True)
    base = PortfolioManager(Mock(), [_strategy("S")], book="baseline", simulate_fills=True)
    base.process_signal(base.strategies["S"], _buy(50_000.0), 50_000.0, 0.6)
    assert db.get_strategy("S")["capital"] == 10_000
