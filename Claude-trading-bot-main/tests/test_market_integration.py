"""The bot collects market data and every strategy sees it as candle columns."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pandas as pd

import config
from market_data import METRICS
from tests.test_market_data import T0, _fake_http


def _bot():
    import main
    from market_data import MarketDataCollector
    bot = main.TradingBot.__new__(main.TradingBot)
    idx = pd.date_range(T0 - timedelta(hours=6), periods=6, freq="1h", tz="UTC")
    candles = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": np.arange(6.0),
                            "volume": 1.0}, index=idx)
    bot.client = Mock()
    bot.client.get_latest_candles.return_value = candles
    bot.client.get_historical_klines.return_value = candles
    strat = Mock(is_active=True, candle_interval="1h")
    bot.strategies, bot._strat_dfs = [strat], {}
    bot.market = MarketDataCollector(config.SYMBOL, http=_fake_http(), clock=lambda: T0)
    return bot


def test_live_candles_carry_the_market_data_columns(temp_db):
    bot = _bot()
    bot.market.update(force=True)
    bot._refresh_candles()
    df = bot._strat_dfs["1h"]
    for m in METRICS:
        assert m in df.columns
    assert df["top_pos_ratio"].notna().any()


def test_learning_history_is_enriched_too(temp_db):
    bot = _bot()
    bot.market.update(force=True)
    df = bot._learning_history("1h", 30, T0 + timedelta(hours=1))
    assert "funding_rate" in df and df["top_pos_ratio"].notna().any()
