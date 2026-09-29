"""A1: the drawdown guard must measure equity, not free capital."""

from unittest.mock import Mock

import pytest

import config
import database as db
from portfolio_manager import PortfolioManager
from strategies.base_strategy import Signal, SignalType


def _client():
    client = Mock()
    client.place_market_order.side_effect = lambda symbol, side, qty: {"orderId": "T"}
    return client


def _strategy(name="RiskTest"):
    strat = Mock()
    strat.name = name
    strat.is_active = True
    return strat


def _buy(price):
    return Signal(SignalType.BUY, 0.7,
                  stop_loss=price * 0.5, take_profit=price * 2.0)


@pytest.fixture
def pm(temp_db, monkeypatch):
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 10_000.0)
    return PortfolioManager(_client(), [_strategy()])


def test_large_position_without_loss_does_not_trigger_guard(pm):
    price = 50_000.0
    assert pm.process_signal(pm.strategies["RiskTest"], _buy(price), price, ml_confidence=0.6)

    pos = db.get_open_positions("RiskTest")[0]
    committed_pct = pos["entry_price"] * pos["quantity"] / 10_000.0
    assert committed_pct > config.MAX_PORTFOLIO_DRAWDOWN_PCT   # >20% of capital committed

    # Price unchanged -> equity unchanged -> a second entry must be allowed.
    assert pm.process_signal(pm.strategies["RiskTest"], _buy(price), price, ml_confidence=0.6)
    assert len(db.get_open_positions("RiskTest")) == 2


def test_real_equity_drawdown_over_limit_triggers_guard(pm):
    entry = 50_000.0
    assert pm.process_signal(pm.strategies["RiskTest"], _buy(entry), entry, ml_confidence=0.6)
    pos = db.get_open_positions("RiskTest")[0]
    notional = pos["entry_price"] * pos["quantity"]

    # Choose a price where the unrealized loss is 21% of the strategy equity.
    loss_needed = 0.21 * 10_000.0
    crash_price = entry * (1 - loss_needed / notional)
    assert crash_price > 0

    assert not pm.process_signal(pm.strategies["RiskTest"], _buy(crash_price),
                                 crash_price, ml_confidence=0.6)
    assert len(db.get_open_positions("RiskTest")) == 1


def test_small_equity_drawdown_does_not_trigger_guard(pm):
    entry = 50_000.0
    assert pm.process_signal(pm.strategies["RiskTest"], _buy(entry), entry, ml_confidence=0.6)
    dip_price = entry * 0.99   # ~0.3% equity loss
    assert pm.process_signal(pm.strategies["RiskTest"], _buy(dip_price),
                             dip_price, ml_confidence=0.6)
