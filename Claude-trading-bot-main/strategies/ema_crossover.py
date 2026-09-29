"""
Strategy 3: EMA Crossover (9 / 21) with 50-EMA Trend Filter
─────────────────────────────────────────────────────────────
Entry LONG  : EMA9 crosses above EMA21 AND close > EMA50
Entry SHORT : EMA9 crosses below EMA21 AND close < EMA50
Exit        : Opposite crossover OR close crosses the EMA50 (invalidating the trend)
Stop-loss   : Below/above the recent swing low/high (approx 2× ATR)
Take-profit : 4× ATR from entry
Best suited for: moderate trends, especially on 1H chart.
"""

import pandas as pd

import config
from .base_strategy import BaseStrategy, ParamSpec, Signal, SignalType


class EMACrossoverStrategy(BaseStrategy):

    SOURCE = "EMA 9/21 crossover with EMA-50 filter (common trend-following setup); repo strategy, previously unregistered"
    TUNABLE_PARAMS = {
        "atr_sl_mult": ParamSpec(min=1.0, max=3.0, step=0.25),
        "atr_tp_mult": ParamSpec(min=2.0, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        defaults = config.STRATEGY_PARAMS.get("EMA_Crossover", {})
        defaults = dict(defaults)   # never mutate the shared config.STRATEGY_PARAMS
        if params:
            defaults.update(params)
        super().__init__("EMA_Crossover", defaults)

    @property
    def min_candles(self) -> int:
        return self.params["trend_ema"] + 30

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1h")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles:
            return Signal(SignalType.HOLD, 0.0)

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        close = float(last["close"])
        atr   = float(last.get("atr_14", close * 0.015))

        ema9_now  = float(last.get("ema_9",  close))
        ema21_now = float(last.get("ema_21", close))
        ema50_now = float(last.get("ema_50", close))
        ema9_prev  = float(prev.get("ema_9",  close))
        ema21_prev = float(prev.get("ema_21", close))

        adx   = float(last.get("adx", 20))
        vol_r = float(last.get("volume_ratio", 1.0))

        # ── Golden Cross (LONG) ────────────────────────────────────────────────
        golden_cross = (ema9_prev <= ema21_prev) and (ema9_now > ema21_now)
        if golden_cross and close > ema50_now:
            # Spread between EMAs as trend strength proxy
            spread = (ema9_now - ema21_now) / (ema21_now + 1e-8)
            confidence = self._calc_confidence(spread, adx, vol_r, long=True)
            sl = close - float(self.params.get("atr_sl_mult", 2.0)) * atr
            tp = close + float(self.params.get("atr_tp_mult", 4.0)) * atr
            return Signal(
                SignalType.BUY, confidence,
                stop_loss=sl, take_profit=tp,
                metadata={"ema9": ema9_now, "ema21": ema21_now,
                          "ema50": ema50_now, "adx": adx, "spread": spread}
            )

        # ── Death Cross (SHORT) ────────────────────────────────────────────────
        death_cross = (ema9_prev >= ema21_prev) and (ema9_now < ema21_now)
        if death_cross and close < ema50_now:
            spread = (ema21_now - ema9_now) / (ema21_now + 1e-8)
            confidence = self._calc_confidence(spread, adx, vol_r, long=False)
            sl = close + float(self.params.get("atr_sl_mult", 2.0)) * atr
            tp = close - float(self.params.get("atr_tp_mult", 4.0)) * atr
            return Signal(
                SignalType.SELL, confidence,
                stop_loss=sl, take_profit=tp,
                metadata={"ema9": ema9_now, "ema21": ema21_now,
                          "ema50": ema50_now, "adx": adx, "spread": spread}
            )

        return Signal(SignalType.HOLD, 0.0)

    @staticmethod
    def _calc_confidence(spread: float, adx: float,
                         vol_ratio: float, long: bool) -> float:
        spread_score  = min(spread / 0.005, 1.0)       # 0.5% spread = full score
        adx_score     = min((adx - 15) / 35.0, 1.0)
        volume_score  = min((vol_ratio - 1.0) / 2.0, 1.0)
        confidence = 0.48 + 0.25 * spread_score + 0.15 * adx_score + 0.12 * volume_score
        return min(confidence, 0.92)
