"""
Free news and fundamentals — no API key, no paid service
────────────────────────────────────────────────────────
  • Crypto headlines (RSS: CoinDesk, Cointelegraph, Decrypt), every hour. Each
    headline is tagged with the coins it mentions, a simple word-list tone
    (-1 negative .. +1 positive) and a SEVERE flag for fundamental risk
    (hack, exploit, lawsuit, delisting, ban, outage, insolvency...). Stored in
    news_items; per coin, market_data gets news_tone (mean tone of the last 24 h)
    and news_count.
  • Macro calendar (ForexFactory weekly JSON): high-impact US events (Fed,
    CPI, jobs...), refreshed every 6 h, in macro_events.
  • Stablecoin supply (DefiLlama, daily since 2017): money waiting on the
    side lines; market_data metric stable_supply (market-wide).

Headlines and the calendar only exist from the first run on (no history): the
replay cannot test them. The word-list tone is crude on purpose — transparent
and free — so the confirmation engine uses it only as one vote among many,
and severe alerts only to veto new long entries.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, List, Optional
from xml.etree import ElementTree

import config
import database as db
from market_data import _store
from utils import coin_of

logger = logging.getLogger(__name__)

RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
}
MACRO_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
STABLES_URL = "https://stablecoins.llama.fi/stablecoincharts/all"

COIN_WORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "ether", "eth"],
    "SOL": ["solana", "sol"],
    "BNB": ["bnb", "binance coin", "bnb chain"],
    "XRP": ["xrp", "ripple"],
    "DOGE": ["dogecoin", "doge"],
    "ADA": ["cardano", "ada"],
}
# Fundamental risk: a new long on that coin is vetoed for a day
SEVERE = ["hack", "hacked", "exploit", "exploited", "drained", "stolen", "breach", "lawsuit", "sues",
          "sued", "charges", "charged", "indicted", "fraud", "ban", "bans", "banned", "delist",
          "delisted", "delisting", "halts", "halted", "suspends", "suspended", "outage", "insolvent",
          "insolvency", "bankrupt", "bankruptcy", "rug pull", "depeg", "depegged"]
NEGATIVE = SEVERE + ["plunge", "plunges", "crash", "crashes", "slump", "slumps", "drop", "drops",
                     "falls", "tumbles", "sell-off", "selloff", "outflows", "bearish", "fear",
                     "warning", "warns", "liquidations", "dump", "dumps", "probe", "investigation"]
POSITIVE = ["surge", "surges", "rally", "rallies", "soars", "jumps", "rises", "gains", "record",
            "inflows", "bullish", "approval", "approved", "approves", "adoption", "partnership",
            "launches", "upgrade", "breakout", "all-time high", "ath", "recovers", "rebounds"]


def _words(text: str, words) -> int:
    return sum(len(re.findall(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", text)) for w in words)


def score_headline(title: str) -> dict:
    t = title.lower()
    pos, neg = _words(t, POSITIVE), _words(t, NEGATIVE)
    coins = [c for c, ws in COIN_WORDS.items() if _words(t, ws)]
    return {"coins": coins, "tone": (pos - neg) / max(pos + neg, 1), "severe": int(_words(t, SEVERE) > 0)}


def _default_http(url: str, params: dict = None, as_text: bool = False):
    import requests
    r = requests.get(url, params=params, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0 (educational paper-trading bot)"})
    r.raise_for_status()
    return r.text if as_text else r.json()


def _ensure_tables():
    conn = db.get_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS news_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, source TEXT NOT NULL,
        title TEXT NOT NULL, link TEXT NOT NULL UNIQUE, coins TEXT NOT NULL,
        tone REAL NOT NULL, severe INTEGER NOT NULL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_ts ON news_items(ts)")
    conn.execute("""CREATE TABLE IF NOT EXISTS macro_events (
        ts TEXT NOT NULL, country TEXT NOT NULL, title TEXT NOT NULL, impact TEXT NOT NULL,
        PRIMARY KEY (ts, country, title))""")
    conn.commit()


def recent_headlines(since: datetime, coin: str = None) -> List[dict]:
    _ensure_tables()
    rows = db.get_conn().execute("SELECT * FROM news_items WHERE ts >= ? ORDER BY ts DESC",
                                 (since.isoformat(),)).fetchall()
    items = [dict(r) for r in rows]
    return [i for i in items if coin is None or coin in i["coins"].split(",")]


def severe_alerts(coin: str, since: datetime) -> List[dict]:
    return [i for i in recent_headlines(since, coin) if i["severe"]]


def macro_events_near(now: datetime, before_min: int = 120, after_min: int = 60) -> List[dict]:
    """High-impact US events from `before_min` minutes ahead to `after_min` minutes ago."""
    _ensure_tables()
    lo = (now - timedelta(minutes=after_min)).isoformat()
    hi = (now + timedelta(minutes=before_min)).isoformat()
    rows = db.get_conn().execute(
        "SELECT * FROM macro_events WHERE country='USD' AND impact='High' AND ts >= ? AND ts <= ? "
        "ORDER BY ts", (lo, hi)).fetchall()
    return [dict(r) for r in rows]


class NewsCollector:

    def __init__(self, symbols, http: Callable = None, clock: Optional[Callable[[], datetime]] = None):
        self.symbols = [symbols] if isinstance(symbols, str) else list(symbols)
        self.http = http or _default_http
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _due(self, key: str, every: timedelta, force: bool) -> bool:
        now = self.clock()
        last = db.get_meta(key)
        if not force and last and now - datetime.fromisoformat(last) < every:
            return False
        db.set_meta(key, now.isoformat())
        return True

    def update(self, force: bool = False) -> dict:
        _ensure_tables()
        got = {}
        if self._due("news:rss:last", timedelta(minutes=config.MARKET_DATA_INTERVAL_MIN), force):
            got["headlines"] = self._safe("headlines", self._headlines)
            got["tone"] = self._safe("tone", self._tone)
        if self._due("news:macro:last", timedelta(hours=6), force):
            got["macro"] = self._safe("macro", self._macro)
        if self._due("news:stables:last", timedelta(hours=24), force):
            got["stablecoins"] = self._safe("stablecoins", self._stablecoins)
        if got:
            logger.info("[news] " + ", ".join(f"{k}+{v}" for k, v in got.items()))
        return got

    @staticmethod
    def _safe(name, fn) -> int:
        try:
            return fn()
        except Exception as e:
            logger.warning(f"[news] {name}: {e}")
            return 0

    def _headlines(self) -> int:
        conn, total = db.get_conn(), 0
        for source, url in RSS_FEEDS.items():
            try:
                root = ElementTree.fromstring(self.http(url, as_text=True).encode("utf-8"))
            except Exception as e:
                logger.warning(f"[news] {source}: {e}")
                continue
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if not title or not link:
                    continue
                try:
                    ts = parsedate_to_datetime(item.findtext("pubDate")).astimezone(timezone.utc)
                except Exception:
                    ts = self.clock()
                s = score_headline(title)
                cur = conn.execute(
                    "INSERT OR IGNORE INTO news_items (ts, source, title, link, coins, tone, severe) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts.isoformat(), source, title, link, ",".join(s["coins"]), s["tone"], s["severe"]))
                total += cur.rowcount
        conn.commit()
        return total

    def _tone(self) -> int:
        now = self.clock()
        items = recent_headlines(now - timedelta(hours=24))
        total = 0
        for sym in self.symbols:
            mine = [i["tone"] for i in items if coin_of(sym) in i["coins"].split(",")]
            total += _store(sym, "news_count", [(now.isoformat(), len(mine))])
            if mine:
                total += _store(sym, "news_tone", [(now.isoformat(), sum(mine) / len(mine))])
        return total

    def _macro(self) -> int:
        rows = []
        for e in self.http(MACRO_URL):
            try:
                ts = datetime.fromisoformat(e["date"]).astimezone(timezone.utc).isoformat()
            except Exception:
                continue
            rows.append((ts, e.get("country", ""), e.get("title", ""), e.get("impact", "")))
        conn = db.get_conn()
        conn.executemany("INSERT OR REPLACE INTO macro_events (ts, country, title, impact) VALUES (?, ?, ?, ?)",
                         rows)
        conn.commit()
        return len(rows)

    def _stablecoins(self) -> int:
        data = self.http(STABLES_URL)
        rows = [(datetime.fromtimestamp(int(d["date"]), tz=timezone.utc).isoformat(),
                 (d.get("totalCirculatingUSD") or {}).get("peggedUSD")) for d in data]
        return _store("ALL", "stable_supply", rows)
