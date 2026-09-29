"""
Free market data — what traders and the crowd are doing
───────────────────────────────────────────────────────
No API key, no paid service. Every MARKET_DATA_INTERVAL_MIN the collector
stores, per symbol, in the market_data table:

  top_pos_ratio  Binance futures TOP TRADERS long/short ratio (by position size)
  top_acc_ratio  top traders long/short ratio (by number of accounts)
  global_ratio   ALL accounts long/short ratio (the crowd)
  taker_ratio    taker buy / sell volume ratio (aggressive buyers vs sellers)
  oi_value       open interest in USD
  funding_rate   perpetual funding rate (per 8 h; > 0 = longs pay shorts)
  fng            crypto Fear & Greed index 0..100 (alternative.me, daily)

Binance only serves the last 30 days of the ratio / taker / open-interest
series: history for them starts accumulating from the first run. Funding (years)
and Fear & Greed (since 2018) are back-filled on the first run.

enrich() adds these as columns to an OHLCV DataFrame with an as-of join on the
candle CLOSE time (index + interval): a closed candle only sees values that
were already published, never later ones. Values older than a per-metric
tolerance are left empty rather than carried forward.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

import config
import database as db

logger = logging.getLogger(__name__)

RATIO_ENDPOINTS = {
    "top_pos_ratio": ("/futures/data/topLongShortPositionRatio", "longShortRatio"),
    "top_acc_ratio": ("/futures/data/topLongShortAccountRatio", "longShortRatio"),
    "global_ratio":  ("/futures/data/globalLongShortAccountRatio", "longShortRatio"),
    "taker_ratio":   ("/futures/data/takerlongshortRatio", "buySellRatio"),
    "oi_value":      ("/futures/data/openInterestHist", "sumOpenInterestValue"),
}
METRICS = tuple(RATIO_ENDPOINTS) + ("funding_rate", "fng")
# How long a published value stays valid for a candle (no stale carry-forward)
TOLERANCE = {**{m: pd.Timedelta(hours=6) for m in RATIO_ENDPOINTS},
             "funding_rate": pd.Timedelta(hours=16), "fng": pd.Timedelta(days=2)}
INTERVAL = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def _default_get(url: str, params: dict = None):
    import requests
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _iso_ms(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


def _store(symbol: str, metric: str, rows) -> int:
    rows = [(symbol, metric, ts, float(v)) for ts, v in rows if v is not None and v != ""]
    if not rows:
        return 0
    conn = db.get_conn()
    conn.executemany("INSERT OR REPLACE INTO market_data (symbol, metric, ts, value) VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


def load_series(symbol: str, metric: str) -> pd.Series:
    sym = "ALL" if metric == "fng" else symbol       # Fear & Greed is market-wide
    rows = db.get_conn().execute(
        "SELECT ts, value FROM market_data WHERE symbol=? AND metric=? ORDER BY ts", (sym, metric)
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.Series([r[1] for r in rows], index=idx, name=metric)


class MarketDataCollector:

    def __init__(self, symbol: str = None, http_get: Callable = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.symbol = symbol or config.SYMBOL
        self.http_get = http_get or _default_get
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def series(self, metric: str) -> pd.Series:
        return load_series(self.symbol, metric)

    def update(self, force: bool = False) -> dict:
        """Fetch what is new; each source fails independently. Returns rows stored."""
        now = self.clock()
        key = f"market:{self.symbol}:last_update"
        last = db.get_meta(key)
        if not force and last and now - datetime.fromisoformat(last) < timedelta(
                minutes=config.MARKET_DATA_INTERVAL_MIN):
            return {}
        db.set_meta(key, now.isoformat())
        stored = {}
        for metric, (path, field) in RATIO_ENDPOINTS.items():
            stored[metric] = self._safe(metric, lambda: self._ratio(metric, path, field))
        stored["funding_rate"] = self._safe("funding_rate", self._funding)
        stored["fng"] = self._safe("fng", self._fear_greed)
        logger.info(f"[market] {self.symbol}: " + ", ".join(f"{k}+{v}" for k, v in stored.items()))
        return stored

    def _safe(self, metric, fn) -> int:
        try:
            return fn()
        except Exception as e:
            logger.warning(f"[market] {self.symbol} {metric}: {e}")
            return 0

    def _ratio(self, metric, path, field) -> int:
        data = self.http_get(config.BINANCE_FUTURES_BASE + path,
                             {"symbol": self.symbol, "period": "1h", "limit": 500})
        return _store(self.symbol, metric, [(_iso_ms(d["timestamp"]), d[field]) for d in data])

    def _funding(self) -> int:
        existing = self.series("funding_rate")
        if len(existing):
            start = int(existing.index[-1].timestamp() * 1000) + 1
        else:
            start = int((self.clock() - timedelta(days=config.MARKET_FUNDING_BACKFILL_DAYS)).timestamp() * 1000)
        total = 0
        for _ in range(20):                               # 1000 rows = ~333 days per page
            page = self.http_get(config.BINANCE_FUTURES_BASE + "/fapi/v1/fundingRate",
                                 {"symbol": self.symbol, "startTime": start, "limit": 1000})
            total += _store(self.symbol, "funding_rate",
                            [(_iso_ms(d["fundingTime"]), d["fundingRate"]) for d in page])
            if len(page) < 1000:
                break
            start = int(page[-1]["fundingTime"]) + 1
        return total

    def _fear_greed(self) -> int:
        full = len(load_series(self.symbol, "fng")) == 0
        data = self.http_get(config.FEAR_GREED_URL, {"limit": 0 if full else 10})["data"]
        return _store("ALL", "fng", [(_iso_ms(int(d["timestamp"]) * 1000), d["value"]) for d in data])


def enrich(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """Copy of df with one column per metric, as of each candle's close time."""
    out = df.copy()
    if df.empty:
        for m in METRICS:
            out[m] = np.nan
        return out
    close_time = (df.index + INTERVAL.get(interval, pd.Timedelta(0))).astype("datetime64[ns, UTC]")
    left = pd.DataFrame({"t": close_time, "pos": np.arange(len(df))})
    for metric in METRICS:
        s = load_series(symbol, metric)
        if s.empty:
            out[metric] = np.nan
            continue
        right = pd.DataFrame({"t": s.index.astype("datetime64[ns, UTC]"), metric: s.to_numpy()})
        merged = pd.merge_asof(left.sort_values("t"), right, on="t", direction="backward",
                               tolerance=TOLERANCE[metric]).sort_values("pos")
        out[metric] = merged[metric].to_numpy()
    return out
