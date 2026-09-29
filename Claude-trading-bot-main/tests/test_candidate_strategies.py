"""Candidate (catalog) strategies: build, declare tunables, and generate signals on real-shaped data."""

import numpy as np
import pandas as pd
import pytest

from strategies import CANDIDATE_STRATEGIES
from strategies.base_strategy import SignalType
from utils.indicators import add_all_indicators


def _prices(n=900, seed=11, freq="4h"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    # alternating trends + ranges so every kind of strategy gets a chance to fire
    drift = np.repeat(rng.choice([-0.004, 0.0, 0.004], size=n // 60 + 1), 60)[:n]
    close = 30_000 * np.exp(np.cumsum(drift + rng.normal(0, 0.012, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.006, n)) * close
    vol = rng.lognormal(3, 0.6, n)
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick,
                       "low": np.minimum(open_, close) - wick, "close": close,
                       "volume": vol}, index=idx)
    df = add_all_indicators(df)
    # synthetic market-data columns (see market_data.enrich) for the modern strategies
    t = np.arange(n)
    df["fng"] = 50 + 45 * np.sin(t / 25)
    df["funding_rate"] = 0.0003 + 0.0009 * np.sin(t / 15)
    df["top_pos_ratio"] = 1.5 + np.cumsum(rng.normal(0, 0.05, n))
    df["global_ratio"] = 1.4 + np.cumsum(rng.normal(0, 0.05, n))
    df["top_acc_ratio"] = 1.3 + np.cumsum(rng.normal(0, 0.03, n))
    df["taker_ratio"] = 1.0 + rng.normal(0, 0.1, n)
    df["oi_value"] = 7e9 * (1 + np.cumsum(rng.normal(0, 0.01, n)))
    return df


def test_catalog_is_not_empty_and_names_are_unique():
    from strategies import ALL_STRATEGIES
    names = [S().name for S in ALL_STRATEGIES + CANDIDATE_STRATEGIES]
    assert len(CANDIDATE_STRATEGIES) >= 9
    assert len(names) == len(set(names))


@pytest.mark.parametrize("S", CANDIDATE_STRATEGIES, ids=lambda S: S.__name__)
def test_candidate_builds_with_valid_tunables_and_a_source(S):
    s = S()
    assert s.TUNABLE_PARAMS
    for name, spec in s.TUNABLE_PARAMS.items():
        assert spec.min <= s.params[name] <= spec.max, name
    assert s.SOURCE, f"{s.name} must cite where the strategy comes from"


@pytest.mark.parametrize("S", CANDIDATE_STRATEGIES, ids=lambda S: S.__name__)
def test_candidate_generates_valid_signals_over_history(S):
    s = S()
    df = _prices(freq={"1h": "1h", "4h": "4h", "1d": "1D"}[s.candle_interval])
    fired = 0
    for i in range(s.min_candles, len(df)):
        sig = s.generate_signal(df.iloc[max(0, i - 600): i + 1])
        if sig.type == SignalType.HOLD:
            continue
        fired += 1
        close = float(df["close"].iloc[i])
        assert 0 < sig.confidence <= 1
        if sig.type == SignalType.BUY:
            assert sig.stop_loss < close < sig.take_profit
        else:
            assert sig.take_profit < close < sig.stop_loss
    assert fired > 0, f"{s.name} never produced a signal"


def test_breakout_cooldown_does_not_block_forever():
    """Breakout used len(df) as a clock; with a fixed-size window it never fired again."""
    from strategies.breakout import BreakoutStrategy
    s = BreakoutStrategy()
    df = _prices(freq="4h", seed=5)
    fired = sum(s.generate_signal(df.iloc[i - 100: i + 1]).type != SignalType.HOLD
                for i in range(100, len(df)))
    assert fired >= 2
