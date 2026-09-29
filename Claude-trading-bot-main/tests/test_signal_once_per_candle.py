"""A1b: a signal on a closed candle is acted on once, even across restarts."""

import pytest

import config
import database as db
from portfolio_manager import PortfolioManager
from tests.test_portfolio_risk import _buy, _client, _strategy


@pytest.fixture
def pm(temp_db, monkeypatch):
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 10_000.0)
    return PortfolioManager(_client(), [_strategy()])


def test_same_candle_signal_is_processed_once(pm):
    strat, price = pm.strategies["RiskTest"], 50_000.0
    candle = "2026-09-28 00:00:00+00:00"
    assert pm.process_signal(strat, _buy(price), price, 0.6, candle_ts=candle)
    assert not pm.process_signal(strat, _buy(price), price, 0.6, candle_ts=candle)
    assert len(db.get_open_positions("RiskTest")) == 1


def test_next_candle_can_trade_again(pm):
    strat, price = pm.strategies["RiskTest"], 50_000.0
    assert pm.process_signal(strat, _buy(price), price, 0.6, candle_ts="2026-09-27 00:00:00+00:00")
    assert pm.process_signal(strat, _buy(price), price, 0.6, candle_ts="2026-09-28 00:00:00+00:00")
    assert len(db.get_open_positions("RiskTest")) == 2


def test_processed_candle_survives_restart(pm):
    strat, price = pm.strategies["RiskTest"], 50_000.0
    candle = "2026-09-28 00:00:00+00:00"
    assert pm.process_signal(strat, _buy(price), price, 0.6, candle_ts=candle)

    restarted = PortfolioManager(_client(), [_strategy()])
    assert not restarted.process_signal(restarted.strategies["RiskTest"], _buy(price),
                                        price, 0.6, candle_ts=candle)
