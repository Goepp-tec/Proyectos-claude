"""A3: profit factor is not trusted (nor enough to activate) with few trades."""

from datetime import datetime
from unittest.mock import Mock

import pandas as pd

import config
from backtester import Backtester, BacktestTrade


def _bt(capital=1_000.0):
    strat = Mock()
    strat.name = "MetricsTest"
    strat.candle_interval = "1d"
    return Backtester(strat, pd.DataFrame({"close": [1.0]}), initial_capital=capital)


def _trade(pnl):
    t = datetime(2026, 1, 1)
    return BacktestTrade(side="LONG", entry_price=100, exit_price=100, quantity=1,
                         pnl=pnl, pnl_pct=pnl / 100, fees=0.0, entry_time=t,
                         exit_time=t, duration_hours=24, exit_reason="TP")


def _metrics(pnls, capital=1_000.0, days=365):
    equity = [capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    return _bt(capital)._compute_metrics([_trade(p) for p in pnls], equity, days)


def test_single_winning_trade_profit_factor_not_reliable():
    r = _metrics([500.0])                       # +50% CAGR, 100% WR, no losses
    assert r.total_trades == 1
    assert not r.pf_reliable
    assert r.profit_factor < 1e6                 # was ~3e9 from dividing by 1e-8
    assert "n/a" in r.summary()
    assert not r.passes_threshold                # too few trades to activate


def test_few_trades_cannot_activate_even_with_great_metrics():
    n = config.MIN_BACKTEST_TRADES - 1
    r = _metrics([60.0, -10.0] * (n // 2) + [60.0] * (n % 2))
    assert r.total_trades < config.MIN_BACKTEST_TRADES
    assert r.cagr >= config.MIN_CAGR_THRESHOLD and r.profit_factor >= config.MIN_PROFIT_FACTOR
    assert not r.passes_threshold


def test_enough_trades_with_good_metrics_passes():
    r = _metrics([60.0, -10.0] * 20)             # 40 trades, PF 6, WR 50%
    assert r.total_trades >= config.MIN_BACKTEST_TRADES
    assert r.pf_reliable
    assert abs(r.profit_factor - 6.0) < 1e-6
    assert r.passes_threshold
