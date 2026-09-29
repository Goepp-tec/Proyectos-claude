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

Top traders beyond Binance:

  okx_top_pos_ratio  OKX top traders (top 5% by position value) long/short
                     ratio by position size — ~60 days of history
  okx_top_acc_ratio  OKX top traders long/short ratio by number of accounts
  okx_global_ratio   OKX all accounts long/short ratio (the crowd)
  hl_top_net         Hyperliquid (on-chain perps, every position is public):
                     net exposure of the month's most profitable wallets,
                     (long USD - short USD) / (long + short), -1..+1
  hl_top_net_count   same by number of wallets (each wallet one vote)
  hl_top_holders     how many of those wallets hold the coin
  cot_am_net         CME futures, CFTC Commitments of Traders (weekly, since
                     2018): asset managers' net position / open interest
  cot_lev_net        same for leveraged funds (hedge funds, CTAs)

Binance only serves the last 30 days of the ratio / taker / open-interest
series and OKX ~60 days: history for them starts accumulating from the first
run; Hyperliquid wallet positions only exist from the first snapshot on.
Funding (years), Fear & Greed (since 2018) and the COT reports (since 2018)
are back-filled on the first run. A COT report is dated Tuesday but only
published on Friday: it is stamped at its release time, never its date.

enrich() adds these as columns to an OHLCV DataFrame with an as-of join on the
candle CLOSE time (index + interval): a closed candle only sees values that
were already published, never later ones. Values older than a per-metric
tolerance are left empty rather than carried forward.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

import config
import database as db
from utils import coin_of

logger = logging.getLogger(__name__)

RATIO_ENDPOINTS = {
    "top_pos_ratio": ("/futures/data/topLongShortPositionRatio", "longShortRatio"),
    "top_acc_ratio": ("/futures/data/topLongShortAccountRatio", "longShortRatio"),
    "global_ratio":  ("/futures/data/globalLongShortAccountRatio", "longShortRatio"),
    "taker_ratio":   ("/futures/data/takerlongshortRatio", "buySellRatio"),
    "oi_value":      ("/futures/data/openInterestHist", "sumOpenInterestValue"),
}
OKX_ENDPOINTS = {
    "okx_top_pos_ratio": "/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader",
    "okx_top_acc_ratio": "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
    "okx_global_ratio":  "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
}
OKX_BACKFILL_DAYS = 60        # OKX keeps the latest 1,440 hourly rows
OKX_PAGE_PAUSE = 0.25         # seconds between pages (rubik limit: 5 requests / 2 s)
HL_METRICS = ("hl_top_net", "hl_top_net_count", "hl_top_holders")
COT_METRICS = ("cot_am_net", "cot_lev_net")
# CFTC contract codes of the CME futures (Traders in Financial Futures report)
CFTC_CODES = {"BTC": "133741", "ETH": "146021"}
COT_RELEASE_DELAY = pd.Timedelta(days=3, hours=21)   # Tue report -> Fri 15:30 New York

METRICS = (tuple(RATIO_ENDPOINTS) + ("funding_rate", "fng") + tuple(OKX_ENDPOINTS)
           + HL_METRICS + COT_METRICS)
# How long a published value stays valid for a candle (no stale carry-forward)
TOLERANCE = {**{m: pd.Timedelta(hours=6) for m in tuple(RATIO_ENDPOINTS) + tuple(OKX_ENDPOINTS)},
             **{m: pd.Timedelta(hours=3) for m in HL_METRICS},
             **{m: pd.Timedelta(days=12) for m in COT_METRICS},
             "funding_rate": pd.Timedelta(hours=16), "fng": pd.Timedelta(days=2)}
INTERVAL = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def _default_http(url: str, params: dict = None, body: dict = None):
    """GET, or POST with a JSON body (Hyperliquid's info API)."""
    import requests
    if body is None:
        r = requests.get(url, params=params, timeout=30)
    else:
        r = requests.post(url, json=body, timeout=30)
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
    # ISO8601: Binance funding times are sometimes '…:00.001' and sometimes ':00'
    idx = pd.to_datetime([r[0] for r in rows], utc=True, format="ISO8601")
    return pd.Series([r[1] for r in rows], index=idx, name=metric)


class MarketDataCollector:
    """
    Collects every source for one or more symbols. Per-symbol sources are
    fetched once per symbol; Fear & Greed and the Hyperliquid wallet snapshot
    (one request per wallet covers every coin) once per update.
    """

    def __init__(self, symbols: Union[str, List[str]] = None, http: Callable = None,
                 clock: Optional[Callable[[], datetime]] = None):
        symbols = symbols or config.SYMBOL
        self.symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        self.symbol = self.symbols[0]
        self.http = http or _default_http
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def series(self, metric: str, symbol: str = None) -> pd.Series:
        return load_series(symbol or self.symbol, metric)

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
        for symbol in self.symbols:
            got = {}
            for metric, (path, field) in RATIO_ENDPOINTS.items():
                got[metric] = self._safe(symbol, metric, lambda: self._ratio(symbol, metric, path, field))
            got["funding_rate"] = self._safe(symbol, "funding_rate", lambda: self._funding(symbol))
            for metric, path in OKX_ENDPOINTS.items():
                got[metric] = self._safe(symbol, metric, lambda: self._okx(symbol, metric, path))
            if coin_of(symbol) in CFTC_CODES:
                got["cot"] = self._safe(symbol, "cot", lambda: self._cot(symbol))
            stored[symbol] = got
        stored["fng"] = self._safe("ALL", "fng", self._fear_greed)
        stored["hyperliquid"] = self._safe("ALL", "hyperliquid", self._hyperliquid)
        for symbol in self.symbols:
            logger.info(f"[market] {symbol}: " + ", ".join(f"{k}+{v}" for k, v in stored[symbol].items()))
        logger.info(f"[market] Fear & Greed +{stored['fng']}, Hyperliquid top wallets +{stored['hyperliquid']}")
        return stored

    def _safe(self, symbol, metric, fn) -> int:
        try:
            return fn()
        except Exception as e:
            logger.warning(f"[market] {symbol} {metric}: {e}")
            return 0

    # ── Binance futures ──────────────────────────────────────────────────────

    def _ratio(self, symbol, metric, path, field) -> int:
        data = self.http(config.BINANCE_FUTURES_BASE + path,
                         {"symbol": symbol, "period": "1h", "limit": 500})
        return _store(symbol, metric, [(_iso_ms(d["timestamp"]), d[field]) for d in data])

    def _funding(self, symbol) -> int:
        existing = load_series(symbol, "funding_rate")
        if len(existing):
            start = int(existing.index[-1].timestamp() * 1000) + 1
        else:
            start = int((self.clock() - timedelta(days=config.MARKET_FUNDING_BACKFILL_DAYS)).timestamp() * 1000)
        total = 0
        for _ in range(20):                               # 1000 rows = ~333 days per page
            page = self.http(config.BINANCE_FUTURES_BASE + "/fapi/v1/fundingRate",
                             {"symbol": symbol, "startTime": start, "limit": 1000})
            total += _store(symbol, "funding_rate",
                            [(_iso_ms(d["fundingTime"]), d["fundingRate"]) for d in page])
            if len(page) < 1000:
                break
            start = int(page[-1]["fundingTime"]) + 1
        return total

    def _fear_greed(self) -> int:
        full = len(load_series("ALL", "fng")) == 0
        data = self.http(config.FEAR_GREED_URL, {"limit": 0 if full else 10})["data"]
        return _store("ALL", "fng", [(_iso_ms(int(d["timestamp"]) * 1000), d["value"]) for d in data])

    # ── OKX top traders ──────────────────────────────────────────────────────

    def _okx(self, symbol, metric, path) -> int:
        """Pages backwards from now until the newest stored row (or ~60 days)."""
        existing = load_series(symbol, metric)
        now_ms = int(self.clock().timestamp() * 1000)
        cutoff = now_ms - OKX_BACKFILL_DAYS * 86_400_000
        known = int(existing.index[-1].timestamp() * 1000) if len(existing) else cutoff
        end, total = now_ms, 0
        for page_no in range(20):                         # 100 hourly rows per page
            if page_no:
                time.sleep(OKX_PAGE_PAUSE)
            data = self.http(config.OKX_BASE + path, {"instId": f"{coin_of(symbol)}-USDT-SWAP",
                                                      "period": "1H", "end": end, "limit": 100})["data"]
            rows = [(int(ts), v) for ts, v in data if int(ts) >= cutoff]
            total += _store(symbol, metric, [(_iso_ms(ts), v) for ts, v in rows])
            if len(data) < 100 or not rows or min(ts for ts, _ in rows) <= known:
                break
            end = min(ts for ts, _ in rows)
        return total

    # ── Hyperliquid: positions of the most profitable wallets ───────────────

    def top_wallets(self) -> List[str]:
        """
        The month's most profitable Hyperliquid wallets, refreshed once a day:
        account >= HL_MIN_ACCOUNT_USD, profitable all-time AND this month, and
        not a market maker (monthly volume <= HL_MAX_TURNOVER x account value).
        """
        key = "market:hl:top_wallets"
        saved = db.get_meta(key)
        if saved:
            s = json.loads(saved)
            if self.clock() - datetime.fromisoformat(s["at"]) < timedelta(hours=config.HL_WALLET_REFRESH_HOURS):
                return s["wallets"]
        rows = self.http(config.HL_LEADERBOARD_URL)["leaderboardRows"]   # ~40 MB, once a day
        ranked = []
        for r in rows:
            perf = {w: p for w, p in r["windowPerformances"]}
            account = float(r["accountValue"])
            month, all_time = perf.get("month"), perf.get("allTime")
            if not month or not all_time or account < config.HL_MIN_ACCOUNT_USD:
                continue
            if float(all_time["pnl"]) <= 0 or float(month["pnl"]) <= 0:
                continue
            if float(month["vlm"]) > config.HL_MAX_TURNOVER * account:
                continue
            ranked.append((float(month["pnl"]), r["ethAddress"]))
        del rows
        wallets = [a for _, a in sorted(ranked, reverse=True)[:config.HL_TOP_WALLETS]]
        db.set_meta(key, json.dumps({"at": self.clock().isoformat(), "wallets": wallets}))
        logger.info(f"[market] Hyperliquid: following the top {len(wallets)} wallets of the month")
        return wallets

    def _hyperliquid(self) -> int:
        wallets = self.top_wallets()

        def positions(addr):
            try:
                st = self.http(config.HL_INFO_URL, body={"type": "clearinghouseState", "user": addr})
                return st.get("assetPositions", [])
            except Exception as e:
                logger.debug(f"[market] Hyperliquid wallet {addr}: {e}")
                return []

        with ThreadPoolExecutor(4) as pool:
            books = list(pool.map(positions, wallets))
        coins = {coin_of(s): s for s in self.symbols}
        agg = {c: [0.0, 0.0, 0, 0] for c in coins}        # long $, short $, n long, n short
        for book in books:
            for ap in book:
                p = ap["position"]
                if p["coin"] not in agg:
                    continue
                szi, value = float(p["szi"]), abs(float(p["positionValue"]))
                a = agg[p["coin"]]
                if szi > 0:
                    a[0] += value; a[2] += 1
                elif szi < 0:
                    a[1] += value; a[3] += 1
        ts, total = self.clock().isoformat(), 0
        for coin, (long_usd, short_usd, n_long, n_short) in agg.items():
            rows = {"hl_top_holders": n_long + n_short}
            if n_long + n_short:
                rows["hl_top_net"] = (long_usd - short_usd) / (long_usd + short_usd)
                rows["hl_top_net_count"] = (n_long - n_short) / (n_long + n_short)
            for metric, v in rows.items():
                total += _store(coins[coin], metric, [(ts, v)])
        return total

    # ── CME institutions: CFTC Commitments of Traders ────────────────────────

    def _cot(self, symbol) -> int:
        code = CFTC_CODES[coin_of(symbol)]
        params = {"cftc_contract_market_code": code, "$order": "report_date_as_yyyy_mm_dd",
                  "$limit": 5000}
        existing = load_series(symbol, "cot_am_net")
        if len(existing):                                  # only reports after the last one
            last_report = (existing.index[-1] - COT_RELEASE_DELAY).strftime("%Y-%m-%dT%H:%M:%S")
            params["$where"] = f"report_date_as_yyyy_mm_dd > '{last_report}'"
        total = 0
        for r in self.http(config.CFTC_URL, params):
            oi = float(r["open_interest_all"])
            if oi <= 0:
                continue
            released = (pd.Timestamp(r["report_date_as_yyyy_mm_dd"][:10], tz="UTC")
                        + COT_RELEASE_DELAY).isoformat()
            am = (float(r["asset_mgr_positions_long"]) - float(r["asset_mgr_positions_short"])) / oi
            lev = (float(r["lev_money_positions_long"]) - float(r["lev_money_positions_short"])) / oi
            total += _store(symbol, "cot_am_net", [(released, am)])
            total += _store(symbol, "cot_lev_net", [(released, lev)])
        return total


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
