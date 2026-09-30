"""Risk engine: aggressiveness 1-10, budget cap, daily loss limit, kill switch, close-only."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

import config
import database as db
from portfolio_manager import PortfolioManager
from strategies.base_strategy import Signal, SignalType
from tests.test_portfolio_risk import _strategy

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
PRICE = 50_000.0


def _sig(side=SignalType.BUY, stop_pct=0.02, conf=0.7, price=PRICE):
    sl = price * (1 - stop_pct) if side == SignalType.BUY else price * (1 + stop_pct)
    tp = price * (1 + 3 * stop_pct) if side == SignalType.BUY else price * (1 - 3 * stop_pct)
    return Signal(side, conf, stop_loss=sl, take_profit=tp)


@pytest.fixture
def setup(temp_db, monkeypatch):
    from risk_engine import RiskEngine, RiskSettings
    clock = {"now": T0}

    def make(aggr=10, budget=100.0, funds=1_000.0, mode="trade", allow_short=True, n=3):
        RiskSettings(funds=funds, budget=budget, aggressiveness=aggr, mode=mode,
                     allow_short=allow_short).save()
        risk = RiskEngine("observe", clock=lambda: clock["now"])
        strats = [_strategy(f"S{i}") for i in range(n)]
        pm = PortfolioManager(Mock(), strats, book="observe", simulate_fills=True,
                              capital_base=budget, risk_engine=risk)
        return risk, pm
    return make, clock


# ─── Profiles ────────────────────────────────────────────────────────────────

def test_profile_scales_with_aggressiveness():
    from risk_engine import profile
    low, mid, high = profile(1), profile(5), profile(10)
    for key in ("risk_per_trade", "max_position", "max_open", "max_exposure", "daily_loss", "max_drawdown"):
        assert low[key] < mid[key] < high[key], key
    assert low["min_confidence"] > high["min_confidence"]
    assert low["statuses"] == ("VIABLE",)
    assert "CONDICIONAL" in mid["statuses"] and "EN_PRUEBA" not in mid["statuses"]
    assert "EN_PRUEBA" in high["statuses"] and "DESCARTADA" not in high["statuses"]


def test_effective_settings_are_persisted_at_startup(temp_db, monkeypatch):
    """Settings coming from .env must be visible in the DB (dashboard, daily review)."""
    from risk_engine import RiskSettings
    monkeypatch.setattr(config, "RISK_BUDGET", 100.0)
    monkeypatch.setattr(config, "RISK_AGGRESSIVENESS", 3)
    assert db.get_meta("risk_settings") is None
    s = RiskSettings.load_and_persist()
    assert (s.budget, s.aggressiveness) == (100.0, 3)
    assert db.get_meta("risk_settings") is not None
    RiskSettings(funds=s.funds, budget=50, aggressiveness=4).save()      # dashboard change wins
    assert RiskSettings.load_and_persist().budget == 50


def test_settings_validation_and_roundtrip(temp_db):
    from risk_engine import RiskSettings
    with pytest.raises(ValueError):
        RiskSettings(funds=1_000, budget=1_500, aggressiveness=5).save()   # budget > funds
    with pytest.raises(ValueError):
        RiskSettings(funds=1_000, budget=100, aggressiveness=11).save()
    with pytest.raises(ValueError):
        RiskSettings(funds=1_000, budget=100, aggressiveness=5, mode="yolo").save()
    RiskSettings(funds=1_000, budget=100, aggressiveness=8, mode="close_only", allow_short=False).save()
    s = RiskSettings.load()
    assert (s.funds, s.budget, s.aggressiveness, s.mode, s.allow_short) == (1_000, 100, 8, "close_only", False)


# ─── Sizing and budget ───────────────────────────────────────────────────────

def test_learning_book_capital_is_the_budget(setup):
    make, _ = setup
    risk, pm = make(budget=100.0)
    assert sum(pm.strategy_equity(n, PRICE) for n in pm.strategies) == pytest.approx(100.0)


def test_position_size_comes_from_risk_and_is_capped(setup):
    make, _ = setup
    risk, pm = make(aggr=10, budget=100.0)
    ok, why, notional = risk.check_entry(pm.strategies["S0"], _sig(stop_pct=0.02), PRICE, 0.6, pm)
    # 3% of 100 at risk with a 2% stop would be 150 USD -> capped at 60% of the budget
    assert ok and notional == pytest.approx(60.0)
    risk1, pm1 = make(aggr=1, budget=100.0)
    ok, why, notional = risk1.check_entry(pm1.strategies["S0"], _sig(stop_pct=0.02), PRICE, 0.7, pm1)
    assert ok and notional == pytest.approx(10.0)          # 0.25% risk / 2% stop = 12.5 -> cap 10%


def test_budget_is_never_exceeded(setup):
    make, _ = setup
    risk, pm = make(aggr=10, budget=100.0)
    opened = [pm.process_signal(pm.strategies[f"S{i}"], _sig(), PRICE, 0.6) for i in range(3)]
    assert opened == [True, True, False]                      # 60 + 40, then nothing left
    exposure = sum(p["entry_price"] * p["quantity"] for p in db.get_open_positions(book="observe"))
    assert exposure <= 100.0 * 1.001


def test_too_small_orders_are_rejected(setup):
    make, _ = setup
    risk, pm = make(aggr=1, budget=20.0)                     # 10% of 20 = 2 USD < 5 USD minimum
    ok, why, _ = risk.check_entry(pm.strategies["S0"], _sig(), PRICE, 0.7, pm)
    assert not ok and "minimo" in why


def test_close_only_mode_and_shorts_switch(setup):
    make, _ = setup
    risk, pm = make(mode="close_only")
    ok, why, _ = risk.check_entry(pm.strategies["S0"], _sig(), PRICE, 0.6, pm)
    assert not ok and "solo cierre" in why
    risk, pm = make(allow_short=False)
    ok, why, _ = risk.check_entry(pm.strategies["S0"], _sig(SignalType.SELL), PRICE, 0.6, pm)
    assert not ok and "cortos" in why


def test_low_confidence_needs_more_aggressiveness(setup):
    make, _ = setup
    risk, pm = make(aggr=1)
    assert not risk.check_entry(pm.strategies["S0"], _sig(conf=0.5), PRICE, 0.6, pm)[0]
    risk, pm = make(aggr=10)
    assert risk.check_entry(pm.strategies["S0"], _sig(conf=0.5), PRICE, 0.6, pm)[0]


# ─── Daily loss limit and kill switch ────────────────────────────────────────

def test_daily_loss_limit_blocks_until_next_day(setup):
    make, clock = setup
    risk, pm = make(aggr=5, budget=100.0)                    # daily limit ~4.1% = ~4.1 USD
    risk.status(pm, PRICE)                                   # day starts at P&L 0
    assert pm.process_signal(pm.strategies["S0"], _sig(), PRICE, 0.6)
    crash = PRICE * 0.85                                      # ~15% on the position -> > 4.1 USD loss
    st = risk.status(pm, crash)
    assert st["daily_limit_hit"] and not st["kill_switch"]
    ok, why, _ = risk.check_entry(pm.strategies["S1"], _sig(price=crash), crash, 0.6, pm)
    assert not ok and "diaria" in why
    clock["now"] = T0 + timedelta(days=1)
    risk.status(pm, crash)                                    # new day, new reference
    assert risk.check_entry(pm.strategies["S1"], _sig(price=crash), crash, 0.6, pm)[0]


def test_kill_switch_stops_entries_until_reset(setup):
    make, clock = setup
    risk, pm = make(aggr=1, budget=100.0)                    # max drawdown 5% = 5 USD
    assert pm.process_signal(pm.strategies["S0"], _sig(), PRICE, 0.7)
    pos = db.get_open_positions(book="observe")[0]
    crash = PRICE * (1 - 6.0 / (pos["entry_price"] * pos["quantity"]))   # ~6 USD loss
    st = risk.status(pm, crash)
    assert st["kill_switch"]
    clock["now"] = T0 + timedelta(days=2)                     # a new day does not clear it
    ok, why, _ = risk.check_entry(pm.strategies["S1"], _sig(price=crash), crash, 0.7, pm)
    assert not ok and "freno" in why
    risk.reset_kill_switch()
    risk.status(pm, crash)
    # aggressiveness 1 allows one position per side and a long is still open: test a short
    ok, why, _ = risk.check_entry(pm.strategies["S1"], _sig(SignalType.SELL, price=crash), crash, 0.7, pm)
    assert ok, why


def test_close_all_request_closes_every_position(setup):
    make, _ = setup
    risk, pm = make(aggr=10)
    pm.process_signal(pm.strategies["S0"], _sig(), PRICE, 0.6)
    pm.process_signal(pm.strategies["S1"], _sig(), PRICE, 0.6)
    risk.request_close_all()
    assert risk.close_all_requested()
    closed = pm.close_all_positions(PRICE, reason="MANUAL_CLOSE_ALL")
    risk.clear_close_all()
    assert closed == 2 and db.get_open_positions(book="observe") == []
    assert not risk.close_all_requested()


# ─── Evaluator gate follows the aggressiveness ───────────────────────────────

def test_aggressiveness_decides_which_ratings_may_trade(temp_db):
    from risk_engine import profile
    from strategy_evaluator import StrategyEvaluator
    from tests.test_strategy_evaluator import Toy, _history
    ev = StrategyEvaluator({}, history_fn=None, clock=None)
    df = _history("1d", 400, T0)
    db.upsert_strategy_status("Toy", "EN_PRUEBA", 0.2, [], ["LONG", "SHORT"], "x", {})
    assert not ev.can_trade(Toy(), SignalType.BUY, df, profile(5)["statuses"])[0]
    assert ev.can_trade(Toy(), SignalType.BUY, df, profile(9)["statuses"])[0]
    db.upsert_strategy_status("Toy", "DESCARTADA", 0.0, [], [], "x", {})
    assert not ev.can_trade(Toy(), SignalType.BUY, df, profile(10)["statuses"])[0]


def test_correlated_positions_are_capped_per_direction(setup):
    """Crypto coins move together: in a replay four shorts opened into the same rally
    and all lost. At most half of the open positions may point the same way."""
    from risk_engine import profile
    assert profile(3)["max_same_side"] == 2 and profile(10)["max_same_side"] == 4
    make, _ = setup
    risk, pm = make(aggr=3, budget=100.0, n=4)
    for i in range(2):
        assert pm.process_signal(pm.strategies[f"S{i}"], _sig(stop_pct=0.05), PRICE, 0.7)
    ok, why, _ = risk.check_entry(pm.strategies["S2"], _sig(stop_pct=0.05), PRICE, 0.7, pm)
    assert not ok and "mismo lado" in why and "LARGO" in why
    ok, why, _ = risk.check_entry(pm.strategies["S2"], _sig(SignalType.SELL, stop_pct=0.05), PRICE, 0.7, pm)
    assert ok, why                                              # the other side is still free
