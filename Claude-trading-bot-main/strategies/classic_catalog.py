"""
Classic public trader strategies (candidate catalog)
────────────────────────────────────────────────────
Well-documented setups adapted to BTCUSDT and to this engine, where exits are
stop-loss / take-profit / max-hold (the original exit rules are approximated
with ATR-based targets — noted per strategy). They do NOT trade in the learning
book until strategy_evaluator.py rates them VIABLE or CONDITIONAL.
"""

import numpy as np
import pandas as pd

import config
from .base_strategy import BaseStrategy, ParamSpec, Signal, SignalType


def _defaults(name: str, params: dict = None) -> dict:
    d = dict(config.STRATEGY_PARAMS.get(name, {}))   # never mutate the shared config
    d.update(params or {})
    return d


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _bracket(side: SignalType, close: float, atr: float, sl_mult: float, tp_mult: float):
    if side == SignalType.BUY:
        return close - sl_mult * atr, close + tp_mult * atr
    return close + sl_mult * atr, close - tp_mult * atr


class TurtleBreakoutStrategy(BaseStrategy):
    """
    Turtle Trading System 1: enter on a close beyond the previous N-day high/low.
    Stop = 2 x ATR ("2N"). The original exit (opposite 10-day breakout) is
    approximated with a 4 x ATR target and a 30-day max hold.
    """
    SOURCE = "Richard Dennis / Curtis Faith, 'Way of the Turtle' (2007), System 1 (20-day breakout, 2N stop)"
    TUNABLE_PARAMS = {
        "entry_period": ParamSpec(min=15, max=55, step=5),
        "atr_sl_mult": ParamSpec(min=1.5, max=3.0, step=0.25),
        "atr_tp_mult": ParamSpec(min=2.0, max=8.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("Turtle_Breakout", _defaults("Turtle_Breakout", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["entry_period"]) + 20

    @property
    def max_hold_candles(self) -> int:
        return 30

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1d")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles:
            return Signal(SignalType.HOLD, 0.0)
        n = int(self.params["entry_period"])
        close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        hi = float(df["high"].iloc[-n - 1:-1].max())
        lo = float(df["low"].iloc[-n - 1:-1].min())
        atr = float(_atr(df).iloc[-1])
        side = None
        if close > hi and prev_close <= hi:
            side, dist = SignalType.BUY, (close - hi) / hi
        elif close < lo and prev_close >= lo:
            side, dist = SignalType.SELL, (lo - close) / lo
        if side is None:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, close, atr, float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(0.85, 0.55 + dist * 10), stop_loss=sl, take_profit=tp,
                      metadata={"channel_high": hi, "channel_low": lo, "atr": atr})


class ConnorsRSI2Strategy(BaseStrategy):
    """
    Larry Connors' RSI(2): buy extreme short-term weakness in a long-term uptrend
    (close > SMA-200, RSI(2) < 10), short extreme strength in a downtrend. The
    original exit (close back above SMA-5) is approximated with a 1.5 x ATR
    target and a 5-day max hold.
    """
    SOURCE = "Larry Connors & Cesar Alvarez, 'Short Term Trading Strategies That Work' (2008), RSI(2)"
    TUNABLE_PARAMS = {
        "rsi_low": ParamSpec(min=5, max=20, step=1),
        "rsi_high": ParamSpec(min=80, max=95, step=1),
        "atr_tp_mult": ParamSpec(min=1.0, max=3.0, step=0.25),
    }

    def __init__(self, params: dict = None):
        super().__init__("Connors_RSI2", _defaults("Connors_RSI2", params))

    @property
    def min_candles(self) -> int:
        return 210

    @property
    def max_hold_candles(self) -> int:
        return 5

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1d")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles:
            return Signal(SignalType.HOLD, 0.0)
        c = df["close"]
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
        rsi2 = float((100 - 100 / (1 + gain / (loss + 1e-12))).iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        close = float(c.iloc[-1])
        atr = float(_atr(df).iloc[-1])
        lo, hi = float(self.params["rsi_low"]), float(self.params["rsi_high"])
        if close > sma200 and rsi2 < lo:
            side, conf = SignalType.BUY, 0.55 + 0.3 * (lo - rsi2) / lo
        elif close < sma200 and rsi2 > hi:
            side, conf = SignalType.SELL, 0.55 + 0.3 * (rsi2 - hi) / (100 - hi)
        else:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, close, atr, float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(conf, 0.85), stop_loss=sl, take_profit=tp,
                      metadata={"rsi2": rsi2, "sma200": sma200})


class BollingerSqueezeStrategy(BaseStrategy):
    """
    Bollinger's "Squeeze": when band width is at a multi-period low
    (volatility contraction), trade the first close outside the bands.
    """
    SOURCE = "John Bollinger, 'Bollinger on Bollinger Bands' (2001), The Squeeze"
    TUNABLE_PARAMS = {
        "squeeze_lookback": ParamSpec(min=60, max=180, step=20),
        "atr_sl_mult": ParamSpec(min=1.0, max=3.0, step=0.25),
        "atr_tp_mult": ParamSpec(min=2.0, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("Bollinger_Squeeze", _defaults("Bollinger_Squeeze", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["squeeze_lookback"]) + int(self.params["bb_period"]) + 5

    @property
    def max_hold_candles(self) -> int:
        return 48

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "4h")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles:
            return Signal(SignalType.HOLD, 0.0)
        p, k = int(self.params["bb_period"]), float(self.params["bb_std"])
        c = df["close"]
        mid = c.rolling(p).mean()
        sd = c.rolling(p).std()
        upper, lower = mid + k * sd, mid - k * sd
        width = (upper - lower) / mid
        look = int(self.params["squeeze_lookback"])
        # squeeze on the previous candle: width in the lowest quantile of the lookback
        threshold = width.iloc[-look - 1:-1].quantile(float(self.params["squeeze_quantile"]))
        if not width.iloc[-2] <= threshold:
            return Signal(SignalType.HOLD, 0.0)
        close = float(c.iloc[-1])
        atr = float(_atr(df).iloc[-1])
        if close > float(upper.iloc[-1]):
            side = SignalType.BUY
        elif close < float(lower.iloc[-1]):
            side = SignalType.SELL
        else:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, close, atr, float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, 0.6, stop_loss=sl, take_profit=tp,
                      metadata={"band_width": float(width.iloc[-1]), "squeeze_threshold": float(threshold)})


class SupertrendStrategy(BaseStrategy):
    """
    Supertrend (ATR 10, multiplier 3): trade the flip of the trend line; the
    line itself is the stop-loss, target = atr_tp_mult x ATR.
    """
    SOURCE = "Olivier Seban, Supertrend indicator (ATR 10 x 3), widely used trend-following setup"
    TUNABLE_PARAMS = {
        "atr_period": ParamSpec(min=7, max=14, step=1),
        "multiplier": ParamSpec(min=2.0, max=4.0, step=0.5),
        "atr_tp_mult": ParamSpec(min=2.0, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("Supertrend", _defaults("Supertrend", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["atr_period"]) * 5 + 10

    @property
    def max_hold_candles(self) -> int:
        return 60

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "4h")

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int, mult: float):
        """Returns (line, direction) arrays; direction +1 up-trend, -1 down-trend."""
        h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
        atr = _atr(df, period).to_numpy()
        hl2 = (h + l) / 2
        up, dn = hl2 - mult * atr, hl2 + mult * atr
        line = np.zeros(len(c))
        direction = np.ones(len(c), dtype=int)
        for i in range(1, len(c)):
            up[i] = max(up[i], up[i - 1]) if c[i - 1] > up[i - 1] else up[i]
            dn[i] = min(dn[i], dn[i - 1]) if c[i - 1] < dn[i - 1] else dn[i]
            if c[i] > dn[i - 1]:
                direction[i] = 1
            elif c[i] < up[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
            line[i] = up[i] if direction[i] == 1 else dn[i]
        return line, direction

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles:
            return Signal(SignalType.HOLD, 0.0)
        period, mult = int(self.params["atr_period"]), float(self.params["multiplier"])
        line, direction = self.supertrend(df, period, mult)
        if direction[-1] == direction[-2]:
            return Signal(SignalType.HOLD, 0.0)
        close = float(df["close"].iloc[-1])
        atr = float(_atr(df, period).iloc[-1])
        tp_mult = float(self.params["atr_tp_mult"])
        if direction[-1] == 1:
            side, sl, tp = SignalType.BUY, float(line[-1]), close + tp_mult * atr
        else:
            side, sl, tp = SignalType.SELL, float(line[-1]), close - tp_mult * atr
        if not (min(sl, tp) < close < max(sl, tp)):
            return Signal(SignalType.HOLD, 0.0)
        return Signal(side, 0.6, stop_loss=sl, take_profit=tp, metadata={"supertrend": float(line[-1])})


class GoldenCross50200Strategy(BaseStrategy):
    """SMA-50 crossing the SMA-200 (golden / death cross)."""
    SOURCE = "Classic 50/200-day moving-average golden/death cross (widely cited trend signal)"
    TUNABLE_PARAMS = {
        "fast_period": ParamSpec(min=30, max=70, step=5),
        "atr_sl_mult": ParamSpec(min=1.5, max=4.0, step=0.5),
        "atr_tp_mult": ParamSpec(min=3.0, max=9.0, step=1.0),
    }

    def __init__(self, params: dict = None):
        super().__init__("Golden_Cross_50_200", _defaults("Golden_Cross_50_200", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["slow_period"]) + 5

    @property
    def max_hold_candles(self) -> int:
        return 120

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1d")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles:
            return Signal(SignalType.HOLD, 0.0)
        c = df["close"]
        fast = c.rolling(int(self.params["fast_period"])).mean()
        slow = c.rolling(int(self.params["slow_period"])).mean()
        up_now, up_prev = fast.iloc[-1] > slow.iloc[-1], fast.iloc[-2] > slow.iloc[-2]
        if up_now == up_prev:
            return Signal(SignalType.HOLD, 0.0)
        side = SignalType.BUY if up_now else SignalType.SELL
        close = float(c.iloc[-1])
        atr = float(_atr(df).iloc[-1])
        sl, tp = _bracket(side, close, atr, float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, 0.6, stop_loss=sl, take_profit=tp,
                      metadata={"sma_fast": float(fast.iloc[-1]), "sma_slow": float(slow.iloc[-1])})
