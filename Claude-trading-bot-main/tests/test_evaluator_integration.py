"""The evaluator gates the learning book; the lab book trades all non-discarded strategies."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
import database as db
from portfolio_manager import PortfolioManager
from strategies.base_strategy import SignalType
from tests.test_portfolio_risk import _buy, _client, _strategy


@pytest.fixture(autouse=True)
def _capital(temp_db, monkeypatch):
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 10_000.0)


def test_blocked_signal_is_logged_but_opens_nothing():
    pm = PortfolioManager(Mock(), [_strategy("S")], book="observe", simulate_fills=True)
    placed = pm.process_signal(pm.strategies["S"], _buy(50_000.0), 50_000.0, 0.6,
                               candle_ts="2026-09-29 00:00:00+00:00",
                               blocked_reason="evaluator: EN_PRUEBA")
    assert not placed
    assert db.get_open_positions(book="observe") == []
    sig = db.get_signals(book="observe")[0]
    assert sig["acted"] == 0 and sig["reason"] == "evaluator: EN_PRUEBA"


def _bot_with_gates():
    import main
    from strategy_evaluator import StrategyEvaluator
    bot = main.TradingBot.__new__(main.TradingBot)
    bot.evaluator = StrategyEvaluator({}, history_fn=None, clock=None)
    return bot


def test_learner_gate_follows_the_evaluator_and_lab_gate_only_blocks_discarded():
    from tests.test_strategy_evaluator import Toy, _history
    from datetime import datetime, timezone
    bot = _bot_with_gates()
    df = _history("1d", 400, datetime(2026, 9, 1, tzinfo=timezone.utc))
    sig = SimpleNamespace(type=SignalType.BUY)

    db.upsert_strategy_status("Toy", "EN_PRUEBA", 0.2, [], ["LONG", "SHORT"], "x", {})
    assert "EN_PRUEBA" in bot._learner_gate(Toy(), sig, df)
    assert bot._lab_gate(Toy(), sig, df) is None               # lab keeps testing it

    db.upsert_strategy_status("Toy", "VIABLE", 0.8, ["TRENDING_UP", "TRENDING_DOWN", "RANGING"],
                              ["LONG", "SHORT"], "x", {})
    assert bot._learner_gate(Toy(), sig, df) is None

    db.upsert_strategy_status("Toy", "DESCARTADA", 0.0, [], [], "x", {})
    assert "DESCARTADA" in bot._learner_gate(Toy(), sig, df)
    assert "DESCARTADA" in bot._lab_gate(Toy(), sig, df)


def test_lab_copies_share_params_with_learners_but_not_signal_state():
    import main
    from strategies import CANDIDATE_STRATEGIES
    learners = [S() for S in CANDIDATE_STRATEGIES]
    lab = main.build_lab(learners)
    for learner, copy in zip(learners, lab):
        assert copy is not learner and copy.params is learner.params and copy.is_active
    name, spec = next(iter(learners[0].TUNABLE_PARAMS.items()))
    learners[0].set_tunable_param(name, spec.max)
    assert lab[0].params[name] == spec.max                    # tuned values follow


def test_locked_observation_mode_includes_strategies_added_later():
    import main
    first = [_strategy("A")]
    main.resolve_trading_mode(first, {"A": Mock(passes_threshold=False)}, False, lock=True)
    later = [_strategy("A"), _strategy("NEW")]
    active, book, mode = main.resolve_trading_mode(later, {}, False, lock=True)
    assert mode == "OBSERVE" and [s.name for s in active] == ["A", "NEW"]


def test_tuner_skips_discarded_strategies():
    from adaptive_tuner import AdaptiveTuner
    from tests.test_adaptive_tuner import Clock, ToyStrategy, _history, _metrics_by_k
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.4, 0.10, 20)})
    strat = ToyStrategy()
    db.upsert_strategy_status("Toy", "DESCARTADA", 0.0, [], [], "x", {})
    AdaptiveTuner({"Toy": strat}, _history, lambda b, n: 1000.0, Clock(),
                  evaluate_fn=ev).run_cycle_if_due()
    assert strat.params["k"] == 3 and db.get_learning_audit("Toy") == []
