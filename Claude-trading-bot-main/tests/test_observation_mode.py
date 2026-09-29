"""A4: if no strategy passes the backtest, observe (no positions) unless explicitly allowed."""

from unittest.mock import Mock

import pytest

import config
import database as db
from tests.test_portfolio_risk import _buy, _strategy


def _results(**passes):
    return {name: Mock(passes_threshold=ok) for name, ok in passes.items()}


def test_nothing_passes_defaults_to_observation_mode():
    import main
    strats = [_strategy("A"), _strategy("B")]
    active, book, mode = main.select_trading_mode(strats, _results(A=False, B=False),
                                                  allow_unvalidated=False)
    assert mode == "OBSERVE" and book == "observe"
    assert [s.name for s in active] == ["A", "B"]


def test_nothing_passes_but_explicitly_allowed_trades_all():
    import main
    strats = [_strategy("A"), _strategy("B")]
    active, book, mode = main.select_trading_mode(strats, _results(A=False, B=False),
                                                  allow_unvalidated=True)
    assert mode == "TRADE_UNVALIDATED" and book == "main"
    assert len(active) == 2


def test_only_validated_strategies_trade():
    import main
    strats = [_strategy("A"), _strategy("B")]
    active, book, mode = main.select_trading_mode(strats, _results(A=True, B=False),
                                                  allow_unvalidated=False)
    assert mode == "TRADE" and book == "main"
    assert [s.name for s in active] == ["A"]


def test_observation_book_logs_signals_without_opening_main_positions(temp_db, monkeypatch):
    import main
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 10_000.0)
    client = Mock()
    pm = main.make_portfolio(client, [_strategy("A")], book="observe")
    price = 50_000.0
    assert pm.process_signal(pm.strategies["A"], _buy(price), price, 0.6,
                             candle_ts="2026-09-28 00:00:00+00:00")

    client.place_market_order.assert_not_called()          # no order of any kind
    assert db.get_open_positions() == []                   # main book stays flat
    assert len(db.get_open_positions(book="observe")) == 1 # theoretical position
    sig = db.get_signals(book="observe")
    assert len(sig) == 1 and sig[0]["acted"] == 1 and sig[0]["signal_type"] == "BUY"


def test_allow_unvalidated_flag_is_read_from_env(monkeypatch):
    import importlib
    monkeypatch.setenv("ALLOW_UNVALIDATED_STRATEGIES", "true")
    importlib.reload(config)
    try:
        assert config.ALLOW_UNVALIDATED_STRATEGIES is True
    finally:
        monkeypatch.delenv("ALLOW_UNVALIDATED_STRATEGIES")
        importlib.reload(config)
    assert config.ALLOW_UNVALIDATED_STRATEGIES is False
