"""The bot applies dashboard risk changes: budget re-split, close-all request, aggressiveness gate."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import database as db
from strategies.base_strategy import SignalType
from tests.test_portfolio_risk import _strategy
from tests.test_risk_engine import PRICE, _sig


@pytest.fixture
def bot(temp_db):
    import main
    from risk_engine import RiskEngine, RiskSettings
    RiskSettings(funds=1_000, budget=100, aggressiveness=10).save()
    b = main.TradingBot.__new__(main.TradingBot)
    b.book = "observe"
    b.risk = RiskEngine("observe")
    b.portfolio = main.make_portfolio(Mock(), [_strategy("S0"), _strategy("S1")], "observe",
                                      capital_base=100, risk_engine=b.risk)
    b._current_price = PRICE
    return b


def test_budget_change_from_dashboard_is_applied(bot):
    from risk_engine import RiskSettings
    RiskSettings(funds=1_000, budget=250, aggressiveness=10).save()
    bot._risk_housekeeping(PRICE)
    assert bot.portfolio.capital_base == 250
    assert sum(bot.portfolio.strategy_equity(n, PRICE) for n in bot.portfolio.strategies) == pytest.approx(250)


def test_close_all_request_closes_positions_and_switches_to_close_only(bot):
    from risk_engine import RiskSettings
    assert bot.portfolio.process_signal(bot.portfolio.strategies["S0"], _sig(), PRICE, 0.6)
    bot.risk.request_close_all()
    bot._risk_housekeeping(PRICE)
    assert db.get_open_positions(book="observe") == []
    assert RiskSettings.load().mode == "close_only"
    assert not bot.risk.close_all_requested()


def test_learner_gate_uses_the_aggressiveness(bot):
    from risk_engine import RiskSettings
    from strategy_evaluator import StrategyEvaluator
    from tests.test_strategy_evaluator import Toy, _history
    bot.evaluator = StrategyEvaluator({}, history_fn=None, clock=None)
    df = _history("1d", 400, datetime(2026, 9, 1, tzinfo=timezone.utc))
    db.upsert_strategy_status("Toy", "EN_PRUEBA", 0.2, [], ["LONG", "SHORT"], "x", {})
    sig = SimpleNamespace(type=SignalType.BUY)
    assert bot._learner_gate(Toy(), sig, df) is None                 # aggressiveness 10
    RiskSettings(funds=1_000, budget=100, aggressiveness=3).save()
    assert "EN_PRUEBA" in bot._learner_gate(Toy(), sig, df)          # prudent: VIABLE only
