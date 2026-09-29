"""Free market data (Binance futures positioning, funding, Fear & Greed): collect, store, join."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import database as db

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
MS = lambda t: int(t.timestamp() * 1000)


HL_BOARD = {"leaderboardRows": [
    # address, account value, all-time pnl, month pnl, month volume
    {"ethAddress": "0xw1", "accountValue": "500000", "windowPerformances": [
        ["month", {"pnl": "300000", "roi": "0.6", "vlm": "1000000"}],
        ["allTime", {"pnl": "2000000", "roi": "4", "vlm": "9000000"}]]},
    {"ethAddress": "0xsmall", "accountValue": "50000", "windowPerformances": [      # too small
        ["month", {"pnl": "900000", "roi": "9", "vlm": "100"}],
        ["allTime", {"pnl": "900000", "roi": "9", "vlm": "100"}]]},
    {"ethAddress": "0xloser", "accountValue": "1000000", "windowPerformances": [    # loses overall
        ["month", {"pnl": "800000", "roi": "0.8", "vlm": "100"}],
        ["allTime", {"pnl": "-1000000", "roi": "-1", "vlm": "100"}]]},
    {"ethAddress": "0xmaker", "accountValue": "200000", "windowPerformances": [     # market maker
        ["month", {"pnl": "700000", "roi": "3", "vlm": "200000000"}],
        ["allTime", {"pnl": "500000", "roi": "2", "vlm": "900000000"}]]},
    {"ethAddress": "0xw5", "accountValue": "300000", "windowPerformances": [
        ["month", {"pnl": "50000", "roi": "0.2", "vlm": "3000000"}],
        ["allTime", {"pnl": "1000000", "roi": "3", "vlm": "9000000"}]]},
]}
HL_POSITIONS = {
    "0xw1": [("BTC", "2.0", "200000"), ("ETH", "-10", "30000")],
    "0xw5": [("BTC", "-1.0", "100000")],
}
# CFTC Traders in Financial Futures rows (report dates are Tuesdays)
COT_ROWS = {"133741": [(d, 20000, 5000 + 500 * i, 1000, 4000, 12000)
                       for i, d in enumerate(("2026-09-08", "2026-09-15", "2026-09-22"))],
            "146021": [("2026-09-22", 10000, 2000, 1000, 1000, 3000)]}


def _okx_rows(end_ms, limit, hours, value):
    """OKX rubik answer: newest first, only rows older than `end`, at most `limit`."""
    times = [T0 - timedelta(hours=h) for h in range(1, hours + 1)]
    rows = [[str(MS(t)), value] for t in times if MS(t) < end_ms]
    return {"code": "0", "data": rows[:limit]}


def _fake_http(calls=None, okx_hours=3):
    """Answers like the real endpoints (verified shapes), recording each path.
    GET by default; POST (Hyperliquid info API) when a JSON body is given."""
    calls = calls if calls is not None else []

    def get(url, params=None, body=None):
        calls.append((url, dict(params or {}) if body is None else dict(body)))
        if body is not None:
            assert "api.hyperliquid.xyz/info" in url and body["type"] == "clearinghouseState"
            return {"assetPositions": [
                {"position": {"coin": c, "szi": szi, "positionValue": v}}
                for c, szi, v in HL_POSITIONS.get(body["user"], [])]}
        if "stats-data.hyperliquid.xyz" in url:
            return HL_BOARD
        if "publicreporting.cftc.gov" in url:
            return [{"report_date_as_yyyy_mm_dd": d + "T00:00:00.000", "open_interest_all": str(oi),
                     "asset_mgr_positions_long": str(al), "asset_mgr_positions_short": str(as_),
                     "lev_money_positions_long": str(ll), "lev_money_positions_short": str(ls)}
                    for d, oi, al, as_, ll, ls in COT_ROWS.get(params["cftc_contract_market_code"], [])]
        if "okx.com" in url:
            assert params["instId"].endswith("-USDT-SWAP")
            value = ("0.95" if "position-ratio-contract-top-trader" in url else
                     "1.10" if "account-ratio-contract-top-trader" in url else "1.40")
            return _okx_rows(int(params.get("end", MS(T0) + 1)), int(params.get("limit", 100)),
                             okx_hours, value)
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
    col = MarketDataCollector("BTCUSDT", http=_fake_http(), clock=lambda: T0)
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
    col = MarketDataCollector("BTCUSDT", http=_fake_http(calls), clock=lambda: now["t"])
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

    def flaky(url, params=None, body=None):
        if "alternative.me" in url:
            raise ConnectionError("down")
        return good(url, params, body)
    col = MarketDataCollector("BTCUSDT", http=flaky, clock=lambda: T0)
    col.update(force=True)
    assert len(col.series("top_pos_ratio")) > 0 and len(col.series("fng")) == 0


def test_enrich_joins_without_looking_ahead(temp_db):
    from market_data import MarketDataCollector, enrich
    col = MarketDataCollector("BTCUSDT", http=_fake_http(), clock=lambda: T0)
    col.update(force=True)
    idx = pd.date_range(T0 - timedelta(hours=5), periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": np.arange(5.0)}, index=idx)
    out = enrich(df, "BTCUSDT", "1h")
    # candle opening at T0-3h closes at T0-2h: it may only see data stamped <= T0-2h
    assert out.loc[T0 - timedelta(hours=3), "top_pos_ratio"] == pytest.approx(1.9)
    assert np.isnan(out.loc[T0 - timedelta(hours=5), "top_pos_ratio"])   # before any data
    assert list(df.columns) == ["close"]                                  # input untouched


def test_mixed_timestamp_formats_are_read(temp_db):
    """Real Binance funding times are sometimes 1 ms past the hour (…00.001)."""
    from market_data import _store, load_series
    _store("BTCUSDT", "funding_rate", [("2022-05-14T08:00:00+00:00", 0.0001),
                                        ("2022-05-14T16:00:00.001000+00:00", 0.0002)])
    s = load_series("BTCUSDT", "funding_rate")
    assert len(s) == 2 and s.iloc[-1] == pytest.approx(0.0002)


def test_enrich_without_any_data_adds_nan_columns(temp_db):
    from market_data import METRICS, enrich
    idx = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    out = enrich(pd.DataFrame({"close": [1.0, 2, 3]}, index=idx), "BTCUSDT", "1d")
    for metric in METRICS:
        assert metric in out and out[metric].isna().all()
