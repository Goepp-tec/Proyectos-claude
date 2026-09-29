"""
Utility functions for the trading bot.
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """
    Get current UTC time as timezone-aware datetime.
    
    This replaces the deprecated datetime.utcnow() which will be removed
    in Python 3.12+.
    """
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Get current UTC time as ISO format string."""
    return utc_now().isoformat()


# ─── Several symbols ──────────────────────────────────────────────────────────

def coin_of(symbol: str) -> str:
    """BTCUSDT -> BTC"""
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def symbol_of(strategy) -> str:
    """The symbol a strategy trades (the primary one if it does not say)."""
    import config
    sym = getattr(strategy, "symbol", None)
    return sym if isinstance(sym, str) and sym else config.SYMBOL


def price_of(prices, symbol: str) -> float:
    """prices is {symbol: price}, or a single float (one-symbol callers)."""
    if isinstance(prices, dict):
        return float(prices.get(symbol) or 0.0)
    return float(prices or 0.0)
