"""B2/B4/B6: validated one-step learning with audit and automatic rollback."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
import database as db
from adaptive_tuner import AdaptiveTuner, Metrics
from strategies.base_strategy import BaseStrategy, ParamSpec, Signal, SignalType

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


class ToyStrategy(BaseStrategy):
    TUNABLE_PARAMS = {"k": ParamSpec(min=1, max=5, step=1)}

    def __init__(self, params=None):
        p = {"k": 3, "candle_interval": "1d"}
        p.update(params or {})
        super().__init__("Toy", p)
        self.is_active = True

    min_candles = 5
    candle_interval = "1d"

    def generate_signal(self, df):
        return Signal(SignalType.HOLD, 0.0)


def _history(interval, days, end, symbol=None):
    idx = pd.date_range(end=end, periods=days, freq="1D", tz="UTC")
    return pd.DataFrame({"close": np.linspace(100, 200, len(idx))}, index=idx)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def _make(evaluate, equities=None, strat=None):
    strat = strat or ToyStrategy()
    clock = Clock()
    eq = equities if equities is not None else {"learn": 1000.0, "baseline": 1000.0}
    tuner = AdaptiveTuner(
        learners={strat.name: strat},
        history_fn=_history,
        equity_fn=lambda book, name: eq["baseline" if book == "baseline" else "learn"],
        clock=clock,
        learner_book="main",
        evaluate_fn=evaluate,
    )
    return tuner, strat, clock, eq


def _metrics_by_k(table):
    """table: {(window, k): (pf, dd, trades)} -> evaluate_fn"""
    def evaluate(strategy, df, window):
        pf, dd, n = table.get((window, strategy.params["k"]), (1.0, 0.10, 20))
        return Metrics(trades=n, profit_factor=pf, max_drawdown=dd, total_pnl=0.0)
    return evaluate


@pytest.fixture(autouse=True)
def _db(temp_db):
    yield


def test_improvement_out_of_sample_is_applied_and_audited():
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.4, 0.10, 20)})
    tuner, strat, _, _ = _make(ev)
    tuner.run_cycle_if_due()
    assert strat.params["k"] == 4
    row = db.get_learning_audit("Toy")[0]
    assert row["decision"] == "applied" and row["param"] == "k"
    assert (row["old_value"], row["new_value"]) == (3, 4)
    assert row["metrics_before"]["profit_factor"] == 1.2
    assert row["metrics_after"]["profit_factor"] == 1.4


def test_change_that_only_helps_in_sample_is_rejected():
    ev = _metrics_by_k({("proposal", 4): (2.0, 0.05, 30),       # looks great where proposed
                        ("validation", 3): (1.3, 0.10, 20),
                        ("validation", 4): (1.1, 0.10, 20)})    # worse on unseen data
    tuner, strat, _, _ = _make(ev)
    tuner.run_cycle_if_due()
    assert strat.params["k"] == 3
    row = db.get_learning_audit("Toy")[0]
    assert row["decision"] == "rejected" and "validation" in row["reason"]


def test_too_few_validation_trades_is_rejected():
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (3.0, 0.02, config.LEARNING_MIN_VALIDATION_TRADES - 1)})
    tuner, strat, _, _ = _make(ev)
    tuner.run_cycle_if_due()
    assert strat.params["k"] == 3
    assert "trades" in db.get_learning_audit("Toy")[0]["reason"]


def test_bigger_drawdown_is_rejected_even_with_better_profit_factor():
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.6, 0.10 + config.LEARNING_MAX_DD_WORSENING + 0.01, 20)})
    tuner, strat, _, _ = _make(ev)
    tuner.run_cycle_if_due()
    assert strat.params["k"] == 3
    assert "drawdown" in db.get_learning_audit("Toy")[0]["reason"]


def test_change_that_underperforms_baseline_live_is_rolled_back():
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.4, 0.10, 20)})
    tuner, strat, clock, eq = _make(ev)
    tuner.run_cycle_if_due()
    assert strat.params["k"] == 4

    # During the rollback window the learner loses 5% while the baseline is flat.
    eq["learn"] = 950.0
    clock.t += timedelta(hours=config.LEARNING_ROLLBACK_WINDOW_HOURS + 1)
    tuner.run_cycle_if_due()
    assert strat.params["k"] == 3
    decisions = [r["decision"] for r in db.get_learning_audit("Toy")]
    assert "rollback" in decisions
    assert db.get_pending_learning_changes() == []


def test_change_that_keeps_up_with_baseline_is_kept():
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.4, 0.10, 20)})
    tuner, strat, clock, eq = _make(ev)
    tuner.run_cycle_if_due()
    eq["learn"], eq["baseline"] = 1010.0, 1000.0
    clock.t += timedelta(hours=config.LEARNING_ROLLBACK_WINDOW_HOURS + 1)
    tuner.run_cycle_if_due()
    assert strat.params["k"] >= 4                      # change kept (maybe a newer one on top)
    assert "rollback" not in [r["decision"] for r in db.get_learning_audit("Toy")]
    assert db.get_learning_audit("Toy", decision="applied")[-1]["evaluated"] == 1


def test_validation_window_starts_after_proposal_window_with_exact_warmup():
    seen = {}

    def spy(strategy, df, window):
        seen.setdefault(window, []).append((strategy.min_candles, df.index[strategy.min_candles],
                                            df.index[-1]))
        return Metrics(trades=50, profit_factor=1.0 + strategy.params["k"],
                       max_drawdown=0.05, total_pnl=0.0)

    tuner, _, clock, _ = _make(spy)
    tuner.run_cycle_if_due()
    val_start = clock.t - timedelta(days=config.LEARNING_VALIDATION_DAYS)
    for _, first_decision, last in seen["proposal"]:
        assert first_decision < val_start and last < val_start      # no overlap
    for _, first_decision, _ in seen["validation"]:
        assert first_decision >= val_start


def test_millisecond_candle_index_with_microsecond_clock():
    """Binance candles are datetime64[ms]; the live clock has microseconds."""
    def history_ms(interval, days, end, symbol=None):
        return _history(interval, days, end).set_axis(
            _history(interval, days, end).index.as_unit("ms"))

    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.4, 0.10, 20)})
    strat = ToyStrategy()
    clock = Clock(T0.replace(hour=3, minute=49, second=48, microsecond=123456))
    tuner = AdaptiveTuner({strat.name: strat}, history_ms, lambda b, n: 1000.0, clock,
                          learner_book="main", evaluate_fn=ev)
    tuner.run_cycle_if_due()
    assert db.get_learning_audit("Toy")[0]["decision"] == "applied"


def test_real_backtester_cycle_on_synthetic_prices_stays_in_bounds():
    from strategies.ema5_momentum import EMA5MomentumStrategy
    rng = np.random.default_rng(7)

    def history(interval, days, end, symbol=None):
        idx = pd.date_range(end=end, periods=days, freq="1D", tz="UTC")
        close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx))))
        df = pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98,
                           "close": close, "volume": 1.0}, index=idx)
        from utils.indicators import add_all_indicators
        return add_all_indicators(df)

    strat = EMA5MomentumStrategy()
    strat.is_active = True
    tuner = AdaptiveTuner({strat.name: strat}, history, lambda b, n: 1000.0, Clock(),
                          learner_book="main")
    tuner.run_cycle_if_due()
    row = db.get_learning_audit(strat.name)[0]
    assert row["decision"] in ("applied", "rejected")
    for p, spec in strat.TUNABLE_PARAMS.items():
        assert spec.min <= strat.params[p] <= spec.max


def test_params_never_leave_their_hard_limits():
    # An evaluator that always prefers a bigger k pushes it up to the limit and no further.
    def always_bigger(strategy, df, window):
        k = strategy.params["k"]
        return Metrics(trades=50, profit_factor=1.0 + k, max_drawdown=0.05, total_pnl=0.0)

    tuner, strat, clock, _ = _make(always_bigger)
    for _ in range(12):
        tuner.run_cycle_if_due()
        clock.t += timedelta(hours=config.LEARNING_ROLLBACK_WINDOW_HOURS + 1)
    spec = ToyStrategy.TUNABLE_PARAMS["k"]
    assert strat.params["k"] == spec.max
    for row in db.get_learning_audit("Toy"):
        if row["new_value"] is not None:
            assert spec.min <= row["new_value"] <= spec.max


def test_cycle_interval_and_daily_change_limit():
    def always_bigger(strategy, df, window):
        return Metrics(trades=50, profit_factor=1.0 + strategy.params["k"],
                       max_drawdown=0.05, total_pnl=0.0)

    tuner, strat, clock, _ = _make(always_bigger)
    tuner.run_cycle_if_due()
    tuner.run_cycle_if_due()                        # same moment: not due again
    clock.t += timedelta(hours=1)
    tuner.run_cycle_if_due()                        # 1 h later: still not due
    assert strat.params["k"] == 4
    assert len(db.get_learning_audit("Toy")) == 1


def test_baseline_never_changes():
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.4, 0.10, 20)})
    baseline = ToyStrategy()
    baseline.freeze()
    tuner, learner, clock, eq = _make(ev)
    tuner.run_cycle_if_due()
    eq["learn"] = 900.0
    clock.t += timedelta(hours=config.LEARNING_ROLLBACK_WINDOW_HOURS + 1)
    tuner.run_cycle_if_due()
    assert baseline.params == ToyStrategy().params

    # A frozen (baseline) strategy handed to the tuner is never modified either.
    frozen = ToyStrategy()
    frozen.freeze()
    t2, _, _, _ = _make(ev, strat=frozen)
    t2.run_cycle_if_due()
    assert frozen.params == ToyStrategy().params


def test_learned_params_survive_restart():
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.4, 0.10, 20)})
    tuner, strat, _, _ = _make(ev)
    tuner.run_cycle_if_due()
    fresh = ToyStrategy()
    AdaptiveTuner.restore_learned_params({"Toy": fresh})
    assert fresh.params["k"] == 4


def test_learning_never_calls_an_llm_or_the_network():
    ev = _metrics_by_k({("proposal", 4): (1.5, 0.10, 20), ("validation", 3): (1.2, 0.10, 20),
                        ("validation", 4): (1.4, 0.10, 20)})
    tuner, strat, _, _ = _make(ev)
    with patch("requests.post", side_effect=AssertionError("network")), \
         patch("requests.get", side_effect=AssertionError("network")), \
         patch.object(config, "ANTHROPIC_API_KEY", ""):
        tuner.run_cycle_if_due()
    assert strat.params["k"] == 4
    import adaptive_tuner, inspect
    src = inspect.getsource(adaptive_tuner).lower()
    assert "anthropic" not in src and "claude" not in src
