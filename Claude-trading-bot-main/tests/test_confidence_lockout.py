"""ML confidence: no permanent lock-out after a loss; circuit breaker still pauses, then expires."""

from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from learning_engine import LearningEngine

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _df(at):
    return pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([at]))


def _close(engine, pnl_pct, at):
    engine.on_trade_closed(trade_id=1, strategy_name="S", entry_price=100, exit_price=100,
                           pnl=pnl_pct * 100, pnl_pct=pnl_pct, side="LONG", duration_hours=1,
                           exit_reason="STOP_LOSS" if pnl_pct < 0 else "TAKE_PROFIT",
                           entry_features={}, df=None, closed_at=at.isoformat())


def test_single_loss_does_not_lock_the_strategy_out():
    eng = LearningEngine({})
    _close(eng, -0.02, T0)
    # Used to be 0.0 (win rate of 1 trade) -> below CONFIDENCE_THRESHOLD forever.
    assert eng.get_confidence("S", _df(T0 + timedelta(hours=1))) >= config.CONFIDENCE_THRESHOLD


def test_long_losing_streak_still_pauses_new_entries():
    eng = LearningEngine({})
    for i in range(8):
        _close(eng, -0.02, T0 + timedelta(days=i))
    assert eng.get_confidence("S", _df(T0 + timedelta(days=8))) < config.CONFIDENCE_THRESHOLD


def test_pause_expires_once_losses_age_out_of_the_window():
    eng = LearningEngine({})
    for i in range(8):
        _close(eng, -0.02, T0 + timedelta(days=i))
    later = T0 + timedelta(days=7 + config.CONFIDENCE_LOOKBACK_DAYS + 1)
    assert eng.get_confidence("S", _df(later)) >= config.CONFIDENCE_THRESHOLD


def test_winning_record_raises_confidence_above_prior():
    eng = LearningEngine({})
    for i in range(15):
        _close(eng, +0.02, T0 + timedelta(days=i))
    assert eng.get_confidence("S", _df(T0 + timedelta(days=15))) > 0.55


def test_history_is_rebuilt_from_db_trades_after_restart():
    eng = LearningEngine({})
    trades = [{"strategy_name": "S", "pnl_pct": -0.02,
               "closed_at": (T0 + timedelta(days=i)).isoformat()} for i in range(8)]
    eng.seed_from_trades(trades)
    assert eng.get_confidence("S", _df(T0 + timedelta(days=8))) < config.CONFIDENCE_THRESHOLD
