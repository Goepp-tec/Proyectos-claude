"""Free market data (Binance futures positioning, funding, Fear & Greed): collect, store, join."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import database as db

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
MS = lambda t: int(t.timestamp() * 1000)


def _fake_http(calls=None):
    """Answers like the real endpoints (verified shapes), recording each path."""
    calls = calls if calls is not None else []

    def get(url, params=None):
        calls.append((url, dict(params or {})))
        hours = [T0 - timedelta(hours=h) for h in range(3, 0, -1)]
        if "topLongShortPositionRatio" in url:
            return [{"symbol": "BTCUSDT", "longShortRatio": str(1.8 + i / 10), "timestamp": MS(t)}
                    for i, t in enumerate(hours)]
        if "topLongShortAccountRatio" in url or "globalLongShortAccountRatio" in url:
            return [{"symbol": "BTCUSDT", "longShortRatio": "1.4", "timestamp": MS(t)} for t in hours]
        if "takerlongshortRatio" in url:
            return [{"buySellRatio": "1.1", "timestamp": MS(t)} for t in hours]
        if "openInterestHist" in url:
            return [{"sumOpenInterestValue": "7751019180.7", "timestamp": MS(t)} for t in hours]
        if "fundingRate" in url:
            start = params.get("startTime", 0)
            times = [T0 - timedelta(hours=8 * k) for k in range(6, 0, -1)]
            return [{"fundingRate": "0.0001", "fundingTime": MS(t)} for t in times if MS(t) >= start]
        if "alternative.me" in url:
            return {"data": [{"value": str(20 + d), "timestamp": str(int((T0 - timedelta(days=d)).timestamp()))}
                             for d in range(3)]}
        raise AssertionError(url)
    get.calls = calls
    return get


def test_update_stores_every_metric_once(temp_db):
    from market_data import METRICS, MarketDataCollector
    col = MarketDataCollector("BTCUSDT", http_get=_fake_http(), clock=lambda: T0)
    col.update(force=True)
    col.update(force=True)                                    # idempotent: no duplicates
    for metric in METRICS:
        s = col.series(metric)
        assert len(s) > 0, metric
        assert not s.index.duplicated().any(), metric
    assert col.series("top_pos_ratio").iloc[-1] == pytest.approx(2.0)
    assert col.series("fng").iloc[-1] == pytest.approx(20)


def test_update_is_throttled_to_the_interval(temp_db):
    from market_data import MarketDataCollector
    calls = []
    now = {"t": T0}
    col = MarketDataCollector("BTCUSDT", http_get=_fake_http(calls), clock=lambda: now["t"])
    col.update()
    n = len(calls)
    now["t"] = T0 + timedelta(minutes=10)
    col.update()
    assert len(calls) == n                                    # too soon: nothing fetched
    now["t"] = T0 + timedelta(hours=1, minutes=1)
    col.update()
    assert len(calls) > n


def test_one_failing_source_does_not_stop_the_others(temp_db):
    from market_data import MarketDataCollector
    good = _fake_http()

    def flaky(url, params=None):
        if "alternative.me" in url:
            raise ConnectionError("down")
        return good(url, params)
    col = MarketDataCollector("BTCUSDT", http_get=flaky, clock=lambda: T0)
    col.update(force=True)
    assert len(col.series("top_pos_ratio")) > 0 and len(col.series("fng")) == 0


def test_enrich_joins_without_looking_ahead(temp_db):
    from market_data import MarketDataCollector, enrich
    col = MarketDataCollector("BTCUSDT", http_get=_fake_http(), clock=lambda: T0)
    col.update(force=True)
    idx = pd.date_range(T0 - timedelta(hours=5), periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": np.arange(5.0)}, index=idx)
    out = enrich(df, "BTCUSDT", "1h")
    # candle opening at T0-3h closes at T0-2h: it may only see data stamped <= T0-2h
    assert out.loc[T0 - timedelta(hours=3), "top_pos_ratio"] == pytest.approx(1.9)
    assert np.isnan(out.loc[T0 - timedelta(hours=5), "top_pos_ratio"])   # before any data
    assert list(df.columns) == ["close"]                                  # input untouched


def test_enrich_without_any_data_adds_nan_columns(temp_db):
    from market_data import METRICS, enrich
    idx = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    out = enrich(pd.DataFrame({"close": [1.0, 2, 3]}, index=idx), "BTCUSDT", "1d")
    for metric in METRICS:
        assert metric in out and out[metric].isna().all()
