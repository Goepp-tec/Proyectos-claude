"""Strategies on top traders beyond Binance: multi-venue consensus, Hyperliquid
top wallets, CME institutions (COT)."""

import numpy as np
import pandas as pd
import pytest

from market_data import METRICS
from strategies.base_strategy import SignalType
from tests.test_candidate_strategies import _prices


def _classes():
    from strategies.smart_money_catalog import (COTInstitutionalStrategy, HLTopWalletsFollowStrategy,
                                                SmartMoneyConsensusStrategy)
    return [SmartMoneyConsensusStrategy, HLTopWalletsFollowStrategy, COTInstitutionalStrategy]


def test_they_are_in_the_candidate_catalog():
    from strategies import CANDIDATE_STRATEGIES
    for S in _classes():
        assert S in CANDIDATE_STRATEGIES


@pytest.mark.parametrize("idx", range(3))
def test_never_trade_without_their_data(idx):
    s = _classes()[idx]()
    df = _prices(freq={"1h": "1h", "4h": "4h", "1d": "1D"}[s.candle_interval]).drop(columns=list(METRICS))
    assert all(s.generate_signal(df.iloc[max(0, i - 600): i + 1]).type == SignalType.HOLD
               for i in range(s.min_candles, len(df), 5))
    df[list(METRICS)] = np.nan
    assert all(s.generate_signal(df.iloc[max(0, i - 600): i + 1]).type == SignalType.HOLD
               for i in range(s.min_candles, len(df), 5))


def _flat_positioning(n=400, seed=4):
    df = _prices(n=n, freq="1h", seed=seed)
    rng = np.random.default_rng(seed)
    df["top_pos_ratio"] = 1.5 + rng.normal(0, 0.02, n)
    df["okx_top_pos_ratio"] = 1.0 + rng.normal(0, 0.02, n)
    df["hl_top_net"], df["hl_top_holders"] = 0.0, 12.0
    ema = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    df.iloc[-1, df.columns.get_loc("close")] = ema * 1.02
    df.iloc[-1, df.columns.get_loc("high")] = ema * 1.03
    return df


def test_consensus_follows_when_top_traders_agree_on_several_venues():
    from strategies.smart_money_catalog import SmartMoneyConsensusStrategy
    s = SmartMoneyConsensusStrategy()
    df = _flat_positioning()
    df.iloc[-1, df.columns.get_loc("top_pos_ratio")] = 1.7        # Binance top traders pile in
    df.iloc[-1, df.columns.get_loc("okx_top_pos_ratio")] = 1.2    # OKX top traders too
    df.iloc[-1, df.columns.get_loc("hl_top_net")] = 0.6           # Hyperliquid whales net long
    sig = s.generate_signal(df)
    assert sig.type == SignalType.BUY and sig.metadata["votes"] == 3


def test_consensus_holds_when_one_venue_disagrees():
    from strategies.smart_money_catalog import SmartMoneyConsensusStrategy
    s = SmartMoneyConsensusStrategy()
    df = _flat_positioning()
    df.iloc[-1, df.columns.get_loc("top_pos_ratio")] = 1.7
    df.iloc[-1, df.columns.get_loc("okx_top_pos_ratio")] = 1.2
    df.iloc[-1, df.columns.get_loc("hl_top_net")] = -0.6          # whales are short
    assert s.generate_signal(df).type == SignalType.HOLD


def test_consensus_needs_two_venues():
    from strategies.smart_money_catalog import SmartMoneyConsensusStrategy
    s = SmartMoneyConsensusStrategy()
    df = _flat_positioning().drop(columns=["okx_top_pos_ratio", "hl_top_net"])
    df.iloc[-1, df.columns.get_loc("top_pos_ratio")] = 1.9        # one venue alone is not enough
    assert s.generate_signal(df).type == SignalType.HOLD


def test_hyperliquid_follow_goes_with_a_swing_of_the_top_wallets():
    from strategies.smart_money_catalog import HLTopWalletsFollowStrategy
    s = HLTopWalletsFollowStrategy()
    df = _flat_positioning()
    df.iloc[-7:, df.columns.get_loc("hl_top_net")] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert s.generate_signal(df).type == SignalType.BUY
    few = df.copy()
    few["hl_top_holders"] = 2.0                                    # two wallets are not a crowd
    assert s.generate_signal(few).type == SignalType.HOLD


def _cot_frame(net_by_week, trend_up=True):
    weeks = len(net_by_week)
    n = weeks * 7
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.full(n, 0.002 if trend_up else -0.002)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": 1.0}, index=idx)
    df["cot_am_net"] = np.repeat(net_by_week, 7)
    return df


def test_cot_buys_when_asset_managers_reach_a_26_week_high():
    from strategies.smart_money_catalog import COTInstitutionalStrategy
    s = COTInstitutionalStrategy()
    net = list(0.10 + 0.02 * np.sin(np.arange(39) / 3)) + [0.2]     # new 26-week high
    df = _cot_frame(net)
    df = df.iloc[: len(df) - 6]                                    # the day the report lands
    sig = s.generate_signal(df)
    assert sig.type == SignalType.BUY and sig.metadata["cot_index"] >= 80
    # the next day is not a new crossing: no repeated entry
    assert s.generate_signal(_cot_frame(net).iloc[: len(df) + 1]).type == SignalType.HOLD


def test_cot_sells_a_26_week_low_in_a_downtrend():
    from strategies.smart_money_catalog import COTInstitutionalStrategy
    s = COTInstitutionalStrategy()
    net = list(0.10 + 0.02 * np.sin(np.arange(39) / 3)) + [0.0]
    df = _cot_frame(net, trend_up=False)
    assert s.generate_signal(df.iloc[: len(df) - 6]).type == SignalType.SELL
