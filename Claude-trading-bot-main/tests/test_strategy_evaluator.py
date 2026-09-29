"""Strategy evaluator: effectiveness per window / market regime / side -> status and score."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import config
import database as db
from strategies.base_strategy import BaseStrategy, ParamSpec, Signal, SignalType

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


class Toy(BaseStrategy):
    TUNABLE_PARAMS = {"k": ParamSpec(min=1, max=5, step=1)}

    def __init__(self, params=None, name="Toy"):
        super().__init__(name, {"k": 3, **(params or {})})
        self.is_active = True

    min_candles = 5
    candle_interval = "1d"

    def generate_signal(self, df):
        return Signal(SignalType.HOLD, 0.0)


def _history(interval, days, end, symbol=None):
    idx = pd.date_range(end=end, periods=days, freq="1D", tz="UTC")
    n = len(idx)
    close = np.linspace(100, 200, n)
    # first half ranging (adx low), second half trending up
    adx = np.r_[np.full(n // 2, 12.0), np.full(n - n // 2, 35.0)]
    return pd.DataFrame({"close": close, "ema_50": close * 0.98, "ema_200": close * 0.95,
                         "adx": adx}, index=idx)


def _metrics(n=40, pf=1.0, windows_profitable=2, worst_dd=0.05, regimes=None, sides=None,
             live=None):
    return dict(trades=n, profit_factor=pf, profitable_windows=windows_profitable,
                windows=4, worst_drawdown=worst_dd, expectancy=0.0,
                by_regime=regimes or {}, by_side=sides or {}, live=live or {"trades": 0, "profit_factor": 0})


# ─── Regime classifier ───────────────────────────────────────────────────────

def test_regime_series_labels():
    from strategy_evaluator import regime_series
    idx = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    df = pd.DataFrame({"close": [110, 90, 100], "ema_50": [105, 95, 100],
                       "ema_200": [100, 100, 100], "adx": [30, 30, 10]}, index=idx)
    assert list(regime_series(df)) == ["TRENDING_UP", "TRENDING_DOWN", "RANGING"]


# ─── Classification rules ────────────────────────────────────────────────────

def test_consistent_profitable_strategy_is_viable():
    from strategy_evaluator import classify
    r = classify(_metrics(n=45, pf=1.5, windows_profitable=4, worst_dd=0.06))
    assert r["status"] == "VIABLE" and 0 < r["score"] <= 1


def test_strategy_that_only_works_in_one_regime_is_conditional():
    from strategy_evaluator import classify
    regimes = {"TRENDING_UP": {"trades": 15, "profit_factor": 1.8},
               "RANGING": {"trades": 20, "profit_factor": 0.6}}
    r = classify(_metrics(n=35, pf=0.95, windows_profitable=2, regimes=regimes))
    assert r["status"] == "CONDICIONAL" and r["allowed_regimes"] == ["TRENDING_UP"]


def test_consistently_losing_strategy_is_discarded_with_value_zero():
    from strategy_evaluator import classify
    regimes = {"RANGING": {"trades": 30, "profit_factor": 0.7}}
    r = classify(_metrics(n=60, pf=0.7, windows_profitable=1, regimes=regimes))
    assert r["status"] == "DESCARTADA" and r["score"] == 0.0


def test_few_trades_is_never_discarded_only_in_testing():
    from strategy_evaluator import classify
    r = classify(_metrics(n=8, pf=0.3, windows_profitable=0))
    assert r["status"] == "EN_PRUEBA"


def test_losing_side_is_blocked():
    from strategy_evaluator import classify
    sides = {"LONG": {"trades": 25, "profit_factor": 1.6}, "SHORT": {"trades": 20, "profit_factor": 0.6}}
    r = classify(_metrics(n=45, pf=1.3, windows_profitable=3, sides=sides))
    assert r["allowed_sides"] == ["LONG"]


# ─── Evaluator cycle ─────────────────────────────────────────────────────────

def _fake_backtest(trades_per_window):
    """trades_per_window: function(window_index) -> list[(side, pnl)] spread over the window."""
    calls = {"n": 0}

    def backtest(strategy, df):
        start = df.index[strategy.min_candles]
        w = calls["n"] % config.EVAL_WINDOWS
        calls["n"] += 1
        spec = trades_per_window(w)
        step = max((df.index[-1] - start) / (len(spec) + 1), timedelta(days=1))
        trades = [(start + step * (i + 1), side, pnl) for i, (side, pnl) in enumerate(spec)]
        return trades, 0.05
    backtest.calls = calls
    return backtest


@pytest.fixture
def evaluator(temp_db):
    from strategy_evaluator import StrategyEvaluator

    def make(backtest, strategies=None):
        strategies = strategies or {"Toy": Toy()}
        return StrategyEvaluator(strategies, history_fn=_history, clock=lambda: T0,
                                 backtest_fn=backtest)
    return make


def test_cycle_rates_and_persists_status_and_history(evaluator):
    ev = evaluator(_fake_backtest(lambda w: [("LONG", 20.0), ("LONG", -8.0)] * 6))
    ev.run_cycle_if_due()
    st = db.get_strategy_status("Toy")
    assert st["status"] == "VIABLE"
    assert st["metrics"]["trades"] == 48 and st["metrics"]["profitable_windows"] == 4
    assert set(st["metrics"]["by_regime"]) <= {"TRENDING_UP", "TRENDING_DOWN", "RANGING"}
    assert len(db.get_strategy_evaluations("Toy")) == 1


def test_discarded_strategy_is_never_reevaluated_or_traded(evaluator):
    losing = _fake_backtest(lambda w: [("LONG", -10.0), ("LONG", 3.0)] * 8)
    ev = evaluator(losing)
    ev.run_cycle_if_due()
    assert db.get_strategy_status("Toy")["status"] == "DESCARTADA"
    calls = losing.calls["n"]

    # Much later, even if it would now look great, it stays at value 0 and is not re-run.
    from strategy_evaluator import StrategyEvaluator
    ev2 = StrategyEvaluator({"Toy": Toy()}, history_fn=_history,
                            clock=lambda: T0 + timedelta(days=30),
                            backtest_fn=_fake_backtest(lambda w: [("LONG", 50.0)] * 20))
    ev2.run_cycle_if_due()
    st = db.get_strategy_status("Toy")
    assert st["status"] == "DESCARTADA" and st["score"] == 0.0
    assert losing.calls["n"] == calls
    ok, why = ev2.can_trade(Toy(), SignalType.BUY, _history("1d", 400, T0))
    assert not ok and "DESCARTADA" in why


def test_can_trade_rules(evaluator):
    ev = evaluator(_fake_backtest(lambda w: []))
    df_trend = _history("1d", 400, T0)                  # last candle: TRENDING_UP
    df_range = df_trend.assign(adx=10.0)                # last candle: RANGING
    db.upsert_strategy_status("Toy", "CONDICIONAL", 0.4, ["TRENDING_UP"], ["LONG"], "x", {})
    assert ev.can_trade(Toy(), SignalType.BUY, df_trend)[0]
    assert not ev.can_trade(Toy(), SignalType.BUY, df_range)[0]      # wrong regime
    assert not ev.can_trade(Toy(), SignalType.SELL, df_trend)[0]     # blocked side
    db.upsert_strategy_status("Toy", "EN_PRUEBA", 0.2, [], ["LONG", "SHORT"], "x", {})
    assert not ev.can_trade(Toy(), SignalType.BUY, df_trend)[0]
    db.upsert_strategy_status("Toy", "VIABLE", 0.8, ["TRENDING_UP", "TRENDING_DOWN", "RANGING"],
                              ["LONG", "SHORT"], "x", {})
    assert ev.can_trade(Toy(), SignalType.SELL, df_range)[0]


def test_windows_do_not_overlap_and_start_after_warmup(evaluator):
    seen = []

    def spy(strategy, df):
        seen.append((df.index[strategy.min_candles], df.index[-1]))
        return [], 0.0

    ev = evaluator(spy)
    ev.run_cycle_if_due()
    assert len(seen) == config.EVAL_WINDOWS
    for (s1, e1), (s2, _) in zip(seen, seen[1:]):
        assert e1 < s2
    assert seen[-1][1] < T0
