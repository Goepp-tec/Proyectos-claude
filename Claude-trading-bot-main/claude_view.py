"""
Claude's review, per coin and side, with an expiry
───────────────────────────────────────────────────
A scheduled Claude session (on the user's PC, no API key in the bot) reviews
the latest trades, the daily report, the news and the charts a few times a
day, and leaves its view here (bot_metadata 'claude_view'):

    long / short : 'a_favor' (one more confirmation), 'neutral', or
                   'bloquear' (veto new entries on that side)
    nota         : why, in a few words
    expires_at   : after it (default 8 h) the view is ignored

Claude can never open a trade: the view only adds one vote or a veto to the
confirmation engine. Written with scripts/claude_view.py inside the container.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import database as db

VALUES = ("a_favor", "neutral", "bloquear")
KEY = "claude_view"


def _load() -> dict:
    try:
        return json.loads(db.get_meta(KEY) or "{}")
    except ValueError:
        return {}


def set_view(coin: str, long: str = "neutral", short: str = "neutral", nota: str = "",
             hours: float = 8, now: datetime = None) -> dict:
    for v in (long, short):
        if v not in VALUES:
            raise ValueError(f"'{v}': use one of {', '.join(VALUES)}")
    if not 0 < float(hours) <= 48:
        raise ValueError("hours must be between 0 and 48")
    now = now or datetime.now(timezone.utc)
    views = _load()
    views[coin.upper()] = {"long": long, "short": short, "nota": nota[:200],
                           "updated_at": now.isoformat(),
                           "expires_at": (now + timedelta(hours=float(hours))).isoformat()}
    db.set_meta(KEY, json.dumps(views))
    return views[coin.upper()]


def get_view(coin: str, now: datetime = None) -> Optional[dict]:
    """The coin's current view, or None if there is none or it expired."""
    now = now or datetime.now(timezone.utc)
    v = _load().get(coin.upper())
    if not v or datetime.fromisoformat(v["expires_at"]) <= now:
        return None
    return v


def all_views(now: datetime = None) -> dict:
    now = now or datetime.now(timezone.utc)
    return {c: dict(v, vigente=datetime.fromisoformat(v["expires_at"]) > now) for c, v in _load().items()}
