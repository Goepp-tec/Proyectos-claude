"""Top traders beyond Binance: OKX top traders, Hyperliquid's most profitable
wallets (on-chain positions) and CME institutions (CFTC Commitments of Traders)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import market_data
from tests.test_market_data import T0, _fake_http


def _collector(symbols="BTCUSDT", calls=None, clock=None, **kw):
    from market_data import MarketDataCollector
    return MarketDataCollector(symbols, http=_fake_http(calls, **kw), clock=clock or (lambda: T0))


def test_okx_top_trader_ratios_are_stored(temp_db):
    col = _collector()
    col.update(force=True)
    assert col.series("okx_top_pos_ratio").iloc[-1] == pytest.approx(0.95)
    assert col.series("okx_top_acc_ratio").iloc[-1] == pytest.approx(1.10)
    assert col.series("okx_global_ratio").iloc[-1] == pytest.approx(1.40)


def test_okx_backfills_its_60_days_page_by_page(temp_db, monkeypatch):
    monkeypatch.setattr(market_data, "OKX_PAGE_PAUSE", 0)
    calls = []
    col = _collector(calls=calls, okx_hours=24 * 70)          # OKX keeps ~60 days
    col.update(force=True)
    s = col.series("okx_top_pos_ratio")
    assert len(s) >= 24 * 59
    assert s.index[0] >= T0 - timedelta(days=market_data.OKX_BACKFILL_DAYS + 1)
    # an hour later only the newest page is asked for
    n_before = sum("okx.com" in u for u, _ in calls)
    col.clock = lambda: T0 + timedelta(hours=1)
    col.update(force=True)
    assert sum("okx.com" in u for u, _ in calls) - n_before == 3   # one page per ratio


def test_hyperliquid_follows_the_most_profitable_wallets(temp_db):
    calls = []
    col = _collector(["BTCUSDT", "ETHUSDT"], calls=calls)
    col.update(force=True)
    asked = {b["user"] for u, b in calls if "hyperliquid.xyz/info" in u}
    # too small, overall losers and market makers are left out; ranked by month pnl
    assert asked == {"0xw1", "0xw5"}
    btc = {m: col.series(m, "BTCUSDT").iloc[-1] for m in ("hl_top_net", "hl_top_net_count", "hl_top_holders")}
    assert btc["hl_top_net"] == pytest.approx((200_000 - 100_000) / 300_000)
    assert btc["hl_top_net_count"] == pytest.approx(0.0) and btc["hl_top_holders"] == 2
    assert col.series("hl_top_net", "ETHUSDT").iloc[-1] == pytest.approx(-1.0)


def test_hyperliquid_positions_are_read_once_for_all_symbols(temp_db):
    calls = []
    col = _collector(["BTCUSDT", "ETHUSDT", "SOLUSDT"], calls=calls)
    col.update(force=True)
    per_wallet = [b["user"] for u, b in calls if "hyperliquid.xyz/info" in u]
    assert sorted(per_wallet) == ["0xw1", "0xw5"]
    # nobody holds SOL: holders 0, no net value invented
    assert col.series("hl_top_holders", "SOLUSDT").iloc[-1] == 0
    assert col.series("hl_top_net", "SOLUSDT").empty


def test_hyperliquid_wallet_list_is_refreshed_daily_not_hourly(temp_db):
    calls = []
    now = {"t": T0}
    col = _collector(calls=calls, clock=lambda: now["t"])
    col.update(force=True)
    now["t"] = T0 + timedelta(hours=2)
    col.update(force=True)
    boards = sum("stats-data.hyperliquid.xyz" in u for u, _ in calls)
    positions = sum("hyperliquid.xyz/info" in u for u, _ in calls)
    assert boards == 1 and positions == 4                    # 2 wallets x 2 updates
    now["t"] = T0 + timedelta(hours=25)
    col.update(force=True)
    assert sum("stats-data.hyperliquid.xyz" in u for u, _ in calls) == 2


def test_cot_is_stamped_at_its_friday_release_not_its_tuesday_date(temp_db):
    col = _collector()
    col.update(force=True)
    am = col.series("cot_am_net")
    # report of Tue 2026-09-22 is published Fri 2026-09-25 (after 15:30 New York)
    assert am.index[-1] == pd.Timestamp("2026-09-25 21:00", tz="UTC")
    assert am.iloc[-1] == pytest.approx((6000 - 1000) / 20000)
    assert col.series("cot_lev_net").iloc[-1] == pytest.approx((4000 - 12000) / 20000)


def test_cot_join_does_not_see_a_report_before_it_is_published(temp_db):
    from market_data import enrich
    _collector().update(force=True)
    idx = pd.date_range("2026-09-23", periods=4, freq="1D", tz="UTC")
    out = enrich(pd.DataFrame({"close": np.ones(4)}, index=idx), "BTCUSDT", "1d")
    # candle of 09-24 closes 09-25 00:00: only the 09-15 report (released 09-18) exists
    assert out.loc["2026-09-24", "cot_am_net"].item() == pytest.approx((5500 - 1000) / 20000)
    # candle of 09-25 closes 09-26 00:00: the 09-22 report is out
    assert out.loc["2026-09-25", "cot_am_net"].item() == pytest.approx((6000 - 1000) / 20000)


def test_cot_only_asks_for_coins_with_cme_futures(temp_db):
    calls = []
    col = _collector(["BTCUSDT", "ETHUSDT", "SOLUSDT"], calls=calls)
    col.update(force=True)
    codes = [p["cftc_contract_market_code"] for u, p in calls if "cftc.gov" in u]
    assert sorted(codes) == ["133741", "146021"]
    assert col.series("cot_am_net", "ETHUSDT").iloc[-1] == pytest.approx((2000 - 1000) / 10000)
    assert col.series("cot_am_net", "SOLUSDT").empty


def test_new_sources_fail_independently(temp_db):
    from market_data import MarketDataCollector
    good = _fake_http()

    def no_hyperliquid(url, params=None, body=None):
        if "hyperliquid" in url:
            raise ConnectionError("down")
        return good(url, params, body)
    col = MarketDataCollector("BTCUSDT", http=no_hyperliquid, clock=lambda: T0)
    col.update(force=True)
    assert col.series("hl_top_net").empty
    assert len(col.series("okx_top_pos_ratio")) and len(col.series("cot_am_net"))
    assert len(col.series("top_pos_ratio"))
