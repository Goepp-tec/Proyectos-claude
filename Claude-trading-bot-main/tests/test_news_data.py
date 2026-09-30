"""Free news & fundamentals: crypto headlines (RSS), macro calendar, stablecoin liquidity."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import database as db

T0 = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def _rss(items):
    """RSS 2.0 as the real feeds serve it (title, link, pubDate)."""
    body = "".join(
        f"<item><title><![CDATA[{title}]]></title><link>{link}</link>"
        f"<pubDate>{(T0 - timedelta(hours=h)).strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate></item>"
        for title, link, h in items)
    return f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>{body}</channel></rss>'


FEEDS = {
    "coindesk": _rss([("Bitcoin ETF inflows hit record as BTC rallies", "https://x/1", 2),
                      ("Solana DEX exploited, $40M drained in hack", "https://x/2", 3),
                      ("Ethereum developers schedule upgrade", "https://x/3", 30)]),   # older than 24 h
    "cointelegraph": _rss([("Bitcoin ETF inflows hit record as BTC rallies", "https://x/1", 2),  # duplicate
                           ("XRP slumps as SEC lawsuit drags on", "https://x/4", 5)]),
}
MACRO = [
    {"title": "CPI m/m", "country": "USD", "date": "2026-09-30T08:30:00-04:00", "impact": "High"},
    {"title": "German Retail Sales", "country": "EUR", "date": "2026-09-30T02:00:00-04:00", "impact": "High"},
    {"title": "FOMC Member Speaks", "country": "USD", "date": "2026-09-30T20:00:00-04:00", "impact": "Low"},
]
STABLES = [{"date": str(int((T0 - timedelta(days=d)).timestamp())),
            "totalCirculatingUSD": {"peggedUSD": 300e9 - d * 1e8}} for d in range(60, -1, -1)]


def _http(calls=None, fail=()):
    calls = calls if calls is not None else []

    def get(url, params=None, as_text=False):
        calls.append(url)
        for bad in fail:
            if bad in url:
                raise ConnectionError("down")
        if "coindesk" in url:
            return FEEDS["coindesk"]
        if "cointelegraph" in url:
            return FEEDS["cointelegraph"]
        if "decrypt" in url:
            return _rss([])
        if "faireconomy" in url:
            return MACRO
        if "llama.fi" in url:
            return STABLES
        raise AssertionError(url)
    get.calls = calls
    return get


def _collector(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"), **kw):
    from news_data import NewsCollector
    return NewsCollector(list(symbols), http=_http(**kw), clock=lambda: T0)


def test_headlines_are_stored_once_with_their_coins_and_tone(temp_db):
    from news_data import recent_headlines
    _collector().update(force=True)
    items = recent_headlines(T0 - timedelta(hours=48))
    titles = [i["title"] for i in items]
    assert titles.count("Bitcoin ETF inflows hit record as BTC rallies") == 1       # same link, 2 feeds
    by = {i["title"]: i for i in items}
    assert by["Bitcoin ETF inflows hit record as BTC rallies"]["coins"] == "BTC"
    assert by["Bitcoin ETF inflows hit record as BTC rallies"]["tone"] > 0
    hack = by["Solana DEX exploited, $40M drained in hack"]
    assert hack["coins"] == "SOL" and hack["severe"] == 1 and hack["tone"] < 0
    assert by["XRP slumps as SEC lawsuit drags on"]["severe"] == 1


def test_news_tone_per_coin_uses_the_last_24_hours(temp_db):
    from market_data import load_series
    _collector().update(force=True)
    assert load_series("BTCUSDT", "news_tone").iloc[-1] > 0
    assert load_series("SOLUSDT", "news_tone").iloc[-1] < 0
    assert load_series("ETHUSDT", "news_count").iloc[-1] == 0          # its headline is 30 h old
    assert load_series("ETHUSDT", "news_tone").empty                   # no tone without headlines


def test_severe_alerts_per_coin(temp_db):
    from news_data import severe_alerts
    _collector().update(force=True)
    sol = severe_alerts("SOL", T0 - timedelta(hours=24))
    assert len(sol) == 1 and "drained" in sol[0]["title"]
    assert severe_alerts("BTC", T0 - timedelta(hours=24)) == []


def test_high_impact_usd_events_near_a_moment(temp_db):
    from news_data import macro_events_near
    _collector().update(force=True)
    cpi = datetime(2026, 9, 30, 12, 30, tzinfo=timezone.utc)           # 08:30 New York
    assert [e["title"] for e in macro_events_near(cpi - timedelta(minutes=90))] == ["CPI m/m"]
    assert macro_events_near(cpi + timedelta(hours=3)) == []           # well after
    assert macro_events_near(cpi - timedelta(hours=5)) == []           # well before
    # EUR and low-impact events do not count


def test_stablecoin_supply_is_a_market_wide_daily_series(temp_db):
    from market_data import enrich, load_series
    _collector().update(force=True)
    s = load_series("BTCUSDT", "stable_supply")                        # shared by every coin
    assert len(s) == 61 and s.iloc[-1] == pytest.approx(300e9)
    idx = pd.date_range(T0 - timedelta(days=5), periods=3, freq="1D", tz="UTC")
    out = enrich(pd.DataFrame({"close": np.ones(3)}, index=idx), "SOLUSDT", "1d")
    assert out["stable_supply"].notna().all()


def test_sources_fail_independently_and_are_throttled(temp_db):
    calls = []
    col = _collector(calls=calls, fail=("llama.fi",))
    col.update()
    from market_data import load_series
    assert len(load_series("BTCUSDT", "news_tone")) and load_series("BTCUSDT", "stable_supply").empty
    n = len(calls)
    col.update()                                                        # same moment: throttled
    assert len(calls) == n
