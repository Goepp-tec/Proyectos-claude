"""Confirmation engine: technical, sentiment, fundamental and Claude checks that must
agree before the learning book opens a position."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import database as db
from strategies.base_strategy import Signal, SignalType

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _frame(interval, trend=0.002, n=400, **cols):
    freq = {"1h": "1h", "4h": "4h", "1d": "1D"}[interval]
    step = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}[interval]
    idx = pd.date_range(end=pd.Timestamp(NOW) - step, periods=n, freq=freq)
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(trend + rng.normal(0, 0.004, n)))
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
                       "volume": 1.0}, index=idx)
    for k, v in cols.items():
        df[k] = v
    return df


def _engine(trend=0.002, frames=None, **cols):
    from confirmations import ConfirmationEngine
    frames = frames or {}

    def dfs(sym, iv):
        if (sym, iv) in frames:
            return frames[(sym, iv)]
        return _frame(iv, trend=trend, **cols)
    return ConfirmationEngine(dfs_fn=dfs, symbols=SYMS, clock=lambda: NOW)


def _strat(symbol="BTCUSDT"):
    s = Mock()
    s.name, s.symbol, s.candle_interval = "S", symbol, "1h"
    return s


BUY = Signal(SignalType.BUY, 0.7, stop_loss=90.0, take_profit=120.0)
SELL = Signal(SignalType.SELL, 0.7, stop_loss=110.0, take_profit=80.0)


def test_min_confirmations_follow_the_aggressiveness():
    from risk_engine import profile
    assert profile(1)["min_confirmations"] == profile(3)["min_confirmations"] == 2
    assert profile(5)["min_confirmations"] == 1
    assert profile(9)["min_confirmations"] == 0


def test_an_uptrend_on_every_timeframe_confirms_a_long_not_a_short(temp_db):
    eng = _engine(trend=0.003)
    ok, reason, summary = eng.decide(eng.checks(_strat(), BUY), min_net=2)
    assert ok, reason
    assert "tendencia diaria" in summary["a_favor"] and "tendencia 4h" in summary["a_favor"]
    ok, reason, _ = eng.decide(eng.checks(_strat(), SELL), min_net=2)
    assert not ok and reason.startswith("confirmaciones:") and "amplitud" in reason


def test_a_severe_headline_vetoes_new_longs_on_that_coin(temp_db):
    from news_data import _ensure_tables
    _ensure_tables()
    db.get_conn().execute(
        "INSERT INTO news_items (ts, source, title, link, coins, tone, severe) VALUES (?,?,?,?,?,?,?)",
        ((NOW - timedelta(hours=3)).isoformat(), "coindesk", "Bitcoin exchange hacked", "u", "BTC", -1.0, 1))
    db.get_conn().commit()
    eng = _engine(trend=0.003)
    ok, reason, _ = eng.decide(eng.checks(_strat("BTCUSDT"), BUY), min_net=0)
    assert not ok and "VETO" in reason and "Bitcoin exchange hacked" in reason
    ok, _, _ = eng.decide(eng.checks(_strat("ETHUSDT"), BUY), min_net=0)      # other coins unaffected
    assert ok


def test_a_high_impact_macro_event_vetoes_both_sides(temp_db):
    from news_data import _ensure_tables
    _ensure_tables()
    db.get_conn().execute("INSERT INTO macro_events VALUES (?, 'USD', 'CPI m/m', 'High')",
                          ((NOW + timedelta(minutes=45)).isoformat(),))
    db.get_conn().commit()
    eng = _engine(trend=0.003)
    for sig in (BUY, SELL):
        ok, reason, _ = eng.decide(eng.checks(_strat(), sig), min_net=0)
        assert not ok and "CPI m/m" in reason


def test_extreme_greed_and_expensive_funding_count_against_longs(temp_db):
    eng = _engine(trend=0.003, fng=88.0, funding_rate=0.0009)
    checks = {c.name: c for c in eng.checks(_strat(), BUY)}
    assert checks["euforia / panico"].vote == -1


def test_missing_data_is_neutral_and_needs_real_confirmations(temp_db):
    from confirmations import ConfirmationEngine
    eng = ConfirmationEngine(dfs_fn=lambda sym, iv: None, symbols=SYMS, clock=lambda: NOW)
    checks = eng.checks(_strat(), BUY)
    assert all(c.vote == 0 for c in checks)
    ok, reason, _ = eng.decide(checks, min_net=1)
    assert not ok and "faltan confirmaciones" in reason
    assert eng.decide(checks, min_net=0)[0]


def test_stablecoin_inflows_support_longs(temp_db):
    supply = np.r_[np.full(370, 250e9), np.linspace(250e9, 260e9, 30)]      # +4% in 30 days
    frames = {("BTCUSDT", "1d"): _frame("1d", trend=0.0, stable_supply=supply)}
    eng = _engine(trend=0.0, frames=frames)
    checks = {c.name: c for c in eng.checks(_strat(), BUY)}
    assert checks["liquidez stablecoins"].vote == 1


def test_claude_can_veto_or_support_a_side_until_its_view_expires(temp_db):
    from claude_view import set_view
    set_view("BTC", long="bloquear", short="a_favor", nota="ETF outflows", hours=6, now=NOW)
    eng = _engine(trend=0.003)
    ok, reason, _ = eng.decide(eng.checks(_strat(), BUY), min_net=0)
    assert not ok and "Claude" in reason and "ETF outflows" in reason
    checks = {c.name: c for c in eng.checks(_strat(), SELL)}
    assert checks["revision de Claude"].vote == 1
    later = _engine(trend=0.003)
    later.clock = lambda: NOW + timedelta(hours=7)                          # expired: neutral
    checks = {c.name: c for c in later.checks(_strat(), BUY)}
    assert checks["revision de Claude"].vote == 0


def test_learning_gate_applies_confirmations_after_the_evaluator(temp_db, monkeypatch):
    import main
    from risk_engine import RiskSettings
    RiskSettings(funds=1_000, budget=100, aggressiveness=3).save()
    bot = main.TradingBot.__new__(main.TradingBot)
    bot.evaluator = Mock()
    bot.evaluator.can_trade.return_value = (True, "")
    bot.confirm = _engine(trend=-0.003)                                      # downtrend
    sig = Signal(SignalType.BUY, 0.7, stop_loss=90.0, take_profit=120.0)
    why = bot._learner_gate(_strat(), sig, _frame("1h"))
    assert why.startswith("confirmaciones:")
    bot.confirm = _engine(trend=0.003)
    assert bot._learner_gate(_strat(), sig, _frame("1h")) is None
    assert sig.metadata["confirmaciones"]["neto"] >= 2                        # kept with the trade
    bot.evaluator.can_trade.return_value = (False, "evaluator: EN_PRUEBA")   # evaluator first
    assert bot._learner_gate(_strat(), sig, _frame("1h")) == "evaluator: EN_PRUEBA"


def test_going_against_the_whole_market_is_vetoed(temp_db):
    """A long while most coins are below their daily EMA-50 (or a short while most are
    above) is vetoed, not just outvoted."""
    eng = _engine(trend=-0.003)
    ok, reason, _ = eng.decide(eng.checks(_strat(), BUY), min_net=0)
    assert not ok and "VETO" in reason and "amplitud" in reason
    ok, reason, _ = eng.decide(eng.checks(_strat(), SELL), min_net=0)
    assert ok, reason
