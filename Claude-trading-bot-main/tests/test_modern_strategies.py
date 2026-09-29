"""Modern strategies: crowd / top-trader positioning, funding, Fear & Greed, Carver EWMAC."""

import numpy as np
import pandas as pd
import pytest

from market_data import METRICS
from strategies.base_strategy import SignalType
from tests.test_candidate_strategies import _prices


def _modern():
    from strategies.modern_catalog import (CarverEWMACStrategy, CrowdVsTopTradersStrategy,
                                           FearGreedContrarianStrategy, FundingContrarianStrategy,
                                           TopTradersFollowStrategy)
    return [FearGreedContrarianStrategy, FundingContrarianStrategy, TopTradersFollowStrategy,
            CrowdVsTopTradersStrategy, CarverEWMACStrategy]


def test_modern_strategies_are_in_the_catalog():
    from strategies import CANDIDATE_STRATEGIES
    for S in _modern():
        assert S in CANDIDATE_STRATEGIES


@pytest.mark.parametrize("idx", range(4))
def test_data_strategies_never_trade_without_their_data(idx):
    S = _modern()[idx]                     # the four that need market data
    s = S()
    df = _prices(freq={"1h": "1h", "4h": "4h", "1d": "1D"}[s.candle_interval]).drop(columns=list(METRICS))
    assert all(s.generate_signal(df.iloc[max(0, i - 600): i + 1]).type == SignalType.HOLD
               for i in range(s.min_candles, len(df)))
    df[list(METRICS)] = np.nan              # present but empty (e.g. before data existed)
    assert all(s.generate_signal(df.iloc[max(0, i - 600): i + 1]).type == SignalType.HOLD
               for i in range(s.min_candles, len(df), 7))


def _last(df, **cols):
    df = df.copy()
    for k, v in cols.items():
        df.iloc[-1, df.columns.get_loc(k)] = v
    return df


def test_fear_greed_contrarian_buys_extreme_fear_on_an_up_day():
    from strategies.modern_catalog import FearGreedContrarianStrategy
    s = FearGreedContrarianStrategy()
    df = _prices(freq="1D")
    o = float(df["open"].iloc[-1])
    up = _last(df, fng=8, close=o * 1.02, high=o * 1.03)
    assert s.generate_signal(up).type == SignalType.BUY
    down = _last(df, fng=92, close=o * 0.98, low=o * 0.97)
    assert s.generate_signal(down).type == SignalType.SELL
    neutral = _last(df, fng=50)
    assert s.generate_signal(neutral).type == SignalType.HOLD


def test_funding_contrarian_fades_crowded_longs():
    from strategies.modern_catalog import FundingContrarianStrategy
    s = FundingContrarianStrategy()
    df = _prices(freq="4h")
    ema = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    crowded_long = _last(df, funding_rate=0.0012, close=ema * 0.97, low=ema * 0.96)
    assert s.generate_signal(crowded_long).type == SignalType.SELL
    crowded_short = _last(df, funding_rate=-0.0008, close=ema * 1.03, high=ema * 1.04)
    assert s.generate_signal(crowded_short).type == SignalType.BUY


def test_top_traders_follow_goes_with_a_sharp_long_build_up():
    from strategies.modern_catalog import TopTradersFollowStrategy
    s = TopTradersFollowStrategy()
    df = _prices(freq="1h")
    df["top_pos_ratio"] = 1.5 + np.random.default_rng(1).normal(0, 0.02, len(df))
    df.iloc[-4:, df.columns.get_loc("top_pos_ratio")] = [1.55, 1.65, 1.75, 1.9]  # top traders pile in
    ema = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    df = _last(df, close=ema * 1.02, high=ema * 1.03)
    assert s.generate_signal(df).type == SignalType.BUY


def test_crowd_vs_top_traders_fades_the_crowd():
    from strategies.modern_catalog import CrowdVsTopTradersStrategy
    s = CrowdVsTopTradersStrategy()
    df = _prices(freq="1h")
    rng = np.random.default_rng(2)
    df["global_ratio"] = 1.4 + rng.normal(0, 0.02, len(df))
    df["top_pos_ratio"] = 1.5 + rng.normal(0, 0.02, len(df))
    df.iloc[-1, df.columns.get_loc("global_ratio")] = 1.7      # crowd suddenly very long
    df.iloc[-1, df.columns.get_loc("top_pos_ratio")] = 1.42    # top traders cutting longs
    assert s.generate_signal(df).type == SignalType.SELL


def test_carver_ewmac_trades_trend_crossings_from_prices_only():
    from strategies.modern_catalog import CarverEWMACStrategy
    s = CarverEWMACStrategy()
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    close = np.r_[np.full(300, 100.0) + np.random.default_rng(3).normal(0, 0.5, 300),
                  100 + np.cumsum(np.full(100, 1.5))]                      # flat, then a trend
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                       "volume": 1.0}, index=idx)
    types = [s.generate_signal(df.iloc[: i + 1]).type for i in range(s.min_candles, n)]
    assert SignalType.BUY in types[280 - s.min_candles:]
