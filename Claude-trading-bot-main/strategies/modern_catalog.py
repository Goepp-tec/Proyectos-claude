"""
Modern candidate strategies — positioning, funding, sentiment, systematic books
───────────────────────────────────────────────────────────────────────────────
The first four read the free market-data columns added by market_data.enrich()
(Binance futures top-trader / crowd long-short ratios, funding rate, Fear &
Greed). Without that data — or before it existed — they HOLD: they never trade
blind. The ratio series only exist for the last 30 days at Binance, so those
strategies stay EN_PRUEBA until enough history has been collected.
Like every candidate, they trade in the learning book only once the strategy
evaluator rates them VIABLE / CONDICIONAL.
"""

import numpy as np
import pandas as pd

import config
from .base_strategy import BaseStrategy, ParamSpec, Signal, SignalType
from .classic_catalog import _atr, _bracket, _defaults


def _has(df: pd.DataFrame, col: str, n: int = 1) -> bool:
    return col in df and len(df) >= n and not df[col].iloc[-n:].isna().any()


def _zscore(series: pd.Series, lookback: int) -> float:
    """Last value vs the mean / std of the previous `lookback` values."""
    prev = series.iloc[-lookback - 1:-1]
    sd = float(prev.std())
    return (float(series.iloc[-1]) - float(prev.mean())) / sd if sd > 0 else 0.0


def _ema(close: pd.Series, span: int) -> float:
    return float(close.ewm(span=span, adjust=False).mean().iloc[-1])


class FearGreedContrarianStrategy(BaseStrategy):
    """Buy extreme fear on an up day, sell extreme greed on a down day."""
    SOURCE = ("alternative.me Crypto Fear & Greed Index (free, daily since 2018) — contrarian "
              "sentiment rule: extreme fear/greed with a reversal candle")
    TUNABLE_PARAMS = {
        "fear_max": ParamSpec(min=10, max=30, step=2),
        "greed_min": ParamSpec(min=70, max=90, step=2),
        "atr_tp_mult": ParamSpec(min=1.5, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("FearGreed_Contrarian", _defaults("FearGreed_Contrarian", params))

    min_candles = 20
    max_hold_candles = 20

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1d")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles or not _has(df, "fng"):
            return Signal(SignalType.HOLD, 0.0)
        fng = float(df["fng"].iloc[-1])
        o, c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
        fear, greed = float(self.params["fear_max"]), float(self.params["greed_min"])
        if fng <= fear and c > o:
            side, conf = SignalType.BUY, 0.55 + 0.3 * (fear - fng) / fear
        elif fng >= greed and c < o:
            side, conf = SignalType.SELL, 0.55 + 0.3 * (fng - greed) / (100 - greed)
        else:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, c, float(_atr(df).iloc[-1]), float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(conf, 0.85), stop_loss=sl, take_profit=tp, metadata={"fng": fng})


class FundingContrarianStrategy(BaseStrategy):
    """Fade crowded perpetual positioning once price stops confirming it."""
    SOURCE = ("Binance USD-M perpetual funding rate (public) — crowded-positioning contrarian "
              "rule: very positive funding + price below EMA-20 → short; very negative + above → long")
    TUNABLE_PARAMS = {
        "funding_high": ParamSpec(min=0.0002, max=0.0012, step=0.0001),
        "funding_low": ParamSpec(min=-0.0006, max=0.0, step=0.0001),
        "atr_tp_mult": ParamSpec(min=1.5, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("Funding_Contrarian", _defaults("Funding_Contrarian", params))

    min_candles = 30
    max_hold_candles = 30

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "4h")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles or not _has(df, "funding_rate"):
            return Signal(SignalType.HOLD, 0.0)
        f = float(df["funding_rate"].iloc[-1])
        c = float(df["close"].iloc[-1])
        ema = _ema(df["close"], 20)
        hi, lo = float(self.params["funding_high"]), float(self.params["funding_low"])
        if f >= hi and c < ema:
            side, conf = SignalType.SELL, 0.55 + min(0.3, (f - hi) / max(hi, 1e-9) * 0.3)
        elif f <= lo and c > ema:
            side, conf = SignalType.BUY, 0.55 + min(0.3, (lo - f) / max(abs(lo), 1e-4) * 0.3)
        else:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, c, float(_atr(df).iloc[-1]), float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(conf, 0.85), stop_loss=sl, take_profit=tp, metadata={"funding": f})


class TopTradersFollowStrategy(BaseStrategy):
    """Follow a sharp build-up of Binance's top traders' positions, with price confirmation."""
    SOURCE = ("Binance Futures 'Top Trader Long/Short Ratio (Positions)' — public statistics of "
              "Binance's top traders; follow a sharp position build-up confirmed by price")
    TUNABLE_PARAMS = {
        "z_entry": ParamSpec(min=1.0, max=3.0, step=0.25),
        "lookback": ParamSpec(min=48, max=168, step=24),
        "atr_tp_mult": ParamSpec(min=1.5, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("TopTraders_Follow", _defaults("TopTraders_Follow", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["lookback"]) + 25

    max_hold_candles = 48

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1h")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        lb = int(self.params["lookback"])
        if len(df) < self.min_candles or not _has(df, "top_pos_ratio", lb + 1):
            return Signal(SignalType.HOLD, 0.0)
        x = df["top_pos_ratio"]
        z = _zscore(x, lb)
        c = float(df["close"].iloc[-1])
        ema = _ema(df["close"], 20)
        ze = float(self.params["z_entry"])
        if z >= ze and x.iloc[-1] > x.iloc[-4] and c > ema:
            side = SignalType.BUY
        elif z <= -ze and x.iloc[-1] < x.iloc[-4] and c < ema:
            side = SignalType.SELL
        else:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, c, float(_atr(df).iloc[-1]), float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(0.85, 0.55 + 0.1 * (abs(z) - ze)), stop_loss=sl, take_profit=tp,
                      metadata={"top_pos_ratio": float(x.iloc[-1]), "z": z})


class CrowdVsTopTradersStrategy(BaseStrategy):
    """Fade the crowd when all accounts pile in and top traders go the other way."""
    SOURCE = ("Binance Futures long/short ratios: all accounts (the crowd) vs top traders — "
              "smart-money / crowd divergence, fade the crowd")
    TUNABLE_PARAMS = {
        "z_entry": ParamSpec(min=1.0, max=3.0, step=0.25),
        "lookback": ParamSpec(min=48, max=168, step=24),
        "atr_tp_mult": ParamSpec(min=1.5, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("Crowd_vs_TopTraders", _defaults("Crowd_vs_TopTraders", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["lookback"]) + 25

    max_hold_candles = 48

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1h")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        lb = int(self.params["lookback"])
        if (len(df) < self.min_candles or not _has(df, "global_ratio", lb + 1)
                or not _has(df, "top_pos_ratio", lb + 1)):
            return Signal(SignalType.HOLD, 0.0)
        zc = _zscore(df["global_ratio"], lb)
        zt = _zscore(df["top_pos_ratio"], lb)
        ze = float(self.params["z_entry"])
        if zc >= ze and zt <= 0:
            side = SignalType.SELL          # crowd very long, top traders not with them
        elif zc <= -ze and zt >= 0:
            side = SignalType.BUY           # crowd very short, top traders adding
        else:
            return Signal(SignalType.HOLD, 0.0)
        c = float(df["close"].iloc[-1])
        sl, tp = _bracket(side, c, float(_atr(df).iloc[-1]), float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(0.85, 0.55 + 0.1 * (abs(zc) - ze)), stop_loss=sl, take_profit=tp,
                      metadata={"z_crowd": zc, "z_top": zt})


class CarverEWMACStrategy(BaseStrategy):
    """
    Robert Carver's EWMAC trend rule: forecast = (EMA fast - EMA slow) / daily
    price volatility x forecast scalar (3.75 for 16/64), capped at +-20. The
    book sizes positions continuously from the forecast; here it becomes an
    entry when the forecast crosses +-threshold (default 10 = average strength).
    """
    SOURCE = "Robert Carver, 'Systematic Trading' (2015) — EWMAC 16/64 trend-following rule"
    TUNABLE_PARAMS = {
        "fast_period": ParamSpec(min=8, max=32, step=4),
        "threshold": ParamSpec(min=5, max=15, step=1),
        "atr_tp_mult": ParamSpec(min=2.0, max=8.0, step=0.5),
    }
    FORECAST_SCALAR = 3.75
    FORECAST_CAP = 20.0

    def __init__(self, params: dict = None):
        super().__init__("Carver_EWMAC", _defaults("Carver_EWMAC", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["fast_period"]) * 4 * 2 + 10

    max_hold_candles = 60

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1d")

    def forecast(self, close: pd.Series) -> pd.Series:
        fast = int(self.params["fast_period"])
        ewmac = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=4 * fast, adjust=False).mean()
        vol = close.diff().ewm(span=36, adjust=False).std()
        return (ewmac / vol.replace(0, np.nan) * self.FORECAST_SCALAR).clip(-self.FORECAST_CAP, self.FORECAST_CAP)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles:
            return Signal(SignalType.HOLD, 0.0)
        fc = self.forecast(df["close"])
        now, prev = float(fc.iloc[-1]), float(fc.iloc[-2])
        th = float(self.params["threshold"])
        if now >= th > prev:
            side = SignalType.BUY
        elif now <= -th < prev:
            side = SignalType.SELL
        else:
            return Signal(SignalType.HOLD, 0.0)
        c = float(df["close"].iloc[-1])
        sl, tp = _bracket(side, c, float(_atr(df).iloc[-1]), float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(0.85, 0.5 + abs(now) / 50), stop_loss=sl, take_profit=tp,
                      metadata={"forecast": now})
