"""Closed trades are journaled exactly once, even across restarts."""

from unittest.mock import Mock

import database as db


def _trade(n):
    db.record_trade(strategy_name="S", symbol="BTCUSDT", side="LONG", entry_price=100,
                    exit_price=101, quantity=1, pnl=1, pnl_pct=0.01, fees_paid=0,
                    entry_time=f"2026-09-2{n}T00:00:00+00:00",
                    exit_time=f"2026-09-2{n}T01:00:00+00:00", duration_hours=1,
                    exit_reason="TAKE_PROFIT")


def _bot():
    import main
    bot = main.TradingBot.__new__(main.TradingBot)   # no network client needed
    bot.book, bot.strategies, bot._strat_dfs = "main", [], {}
    bot.learning = Mock()
    return bot


def test_trades_are_journaled_once_across_restarts(temp_db):
    _trade(1)
    _trade(2)
    bot = _bot()
    bot._journal_new_trades()
    assert bot.learning.on_trade_closed.call_count == 2

    restarted = _bot()
    restarted._journal_new_trades()
    assert restarted.learning.on_trade_closed.call_count == 0   # nothing re-journaled

    _trade(3)
    restarted._journal_new_trades()
    assert restarted.learning.on_trade_closed.call_count == 1
    assert restarted.learning.on_trade_closed.call_args.kwargs["trade_id"] == 3
