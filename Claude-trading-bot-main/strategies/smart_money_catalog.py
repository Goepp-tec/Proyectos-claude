"""
Smart-money strategies — what top traders do on several venues
────────────────────────────────────────────────────────────────
They read the market-data columns added by market_data.enrich():

  top_pos_ratio      Binance top traders long/short (positions)   ~30 days of history
  okx_top_pos_ratio  OKX top traders long/short (positions)       ~60 days
  hl_top_net         Hyperliquid: net exposure of the month's most profitable
                     wallets (public on-chain positions)          from the first snapshot
  cot_am_net         CME asset managers' net position (CFTC COT)  weekly since 2018

Without their data they HOLD. The first two can only be judged on live
evidence (the lab book) as history accumulates; the COT one can be
back-tested over years. Like every candidate, they trade in the learning book
only once the strategy evaluator rates them VIABLE / CONDICIONAL.
"""

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, ParamSpec, Signal, SignalType
from .classic_catalog import _atr, _bracket, _defaults
from .modern_catalog import _ema, _has, _zscore

MIN_HL_HOLDERS = 5          # fewer wallets holding the coin is not a signal


class SmartMoneyConsensusStrategy(BaseStrategy):
    """
    Three venues, three votes: Binance top traders and OKX top traders (sharp
    build-up of their long/short position ratio, z-score over `lookback` hours)
    and Hyperliquid's most profitable wallets (net exposure beyond +-hl_net_min).
    Follow when at least two venues agree, none disagrees, and price confirms.
    """
    SOURCE = ("Top traders on three venues — Binance and OKX public top-trader long/short "
              "ratios + on-chain positions of Hyperliquid's most profitable wallets: follow a "
              "consensus of at least two venues with none against")
    TUNABLE_PARAMS = {
        "z_entry": ParamSpec(min=0.5, max=2.5, step=0.25),
        "lookback": ParamSpec(min=48, max=168, step=24),
        "hl_net_min": ParamSpec(min=0.1, max=0.7, step=0.05),
        "atr_tp_mult": ParamSpec(min=1.5, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("SmartMoney_Consensus", _defaults("SmartMoney_Consensus", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["lookback"]) + 25

    max_hold_candles = 48

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1h")

    def votes(self, df: pd.DataFrame) -> dict:
        """+1 / -1 / 0 per venue that has data; venues without data do not vote."""
        lb, ze = int(self.params["lookback"]), float(self.params["z_entry"])
        out = {}
        for venue, col in (("binance", "top_pos_ratio"), ("okx", "okx_top_pos_ratio")):
            if _has(df, col, lb + 1):
                z = _zscore(df[col], lb)
                out[venue] = 1 if z >= ze else -1 if z <= -ze else 0
        if (_has(df, "hl_top_net") and _has(df, "hl_top_holders")
                and df["hl_top_holders"].iloc[-1] >= MIN_HL_HOLDERS):
            net, th = float(df["hl_top_net"].iloc[-1]), float(self.params["hl_net_min"])
            out["hyperliquid"] = 1 if net >= th else -1 if net <= -th else 0
        return out

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.min_candles:
            return Signal(SignalType.HOLD, 0.0)
        v = self.votes(df)
        if len(v) < 2:
            return Signal(SignalType.HOLD, 0.0)
        up, down = sum(x > 0 for x in v.values()), sum(x < 0 for x in v.values())
        c, ema = float(df["close"].iloc[-1]), _ema(df["close"], 20)
        if up >= 2 and down == 0 and c > ema:
            side, n = SignalType.BUY, up
        elif down >= 2 and up == 0 and c < ema:
            side, n = SignalType.SELL, down
        else:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, c, float(_atr(df).iloc[-1]), float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(0.85, 0.55 + 0.1 * (n - 2) + 0.05 * (len(v) - 2)),
                      stop_loss=sl, take_profit=tp, metadata={"votes": n, "venues": v})


class HLTopWalletsFollowStrategy(BaseStrategy):
    """Follow Hyperliquid's most profitable wallets when they swing to one side."""
    SOURCE = ("Hyperliquid public on-chain positions of the month's most profitable wallets "
              "(leaderboard, market makers excluded) — follow a swing of their net exposure")
    TUNABLE_PARAMS = {
        "net_entry": ParamSpec(min=0.2, max=0.8, step=0.05),
        "swing": ParamSpec(min=0.1, max=0.6, step=0.05),
        "swing_hours": ParamSpec(min=3, max=24, step=3),
        "atr_tp_mult": ParamSpec(min=1.5, max=6.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("HL_TopWallets_Follow", _defaults("HL_TopWallets_Follow", params))

    min_candles = 50
    max_hold_candles = 48

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1h")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        h = int(self.params["swing_hours"])
        if (len(df) < self.min_candles or not _has(df, "hl_top_net", h + 1)
                or not _has(df, "hl_top_holders") or df["hl_top_holders"].iloc[-1] < MIN_HL_HOLDERS):
            return Signal(SignalType.HOLD, 0.0)
        net = df["hl_top_net"]
        now, before = float(net.iloc[-1]), float(net.iloc[-1 - h])
        entry, swing = float(self.params["net_entry"]), float(self.params["swing"])
        c, ema = float(df["close"].iloc[-1]), _ema(df["close"], 20)
        if now >= entry and now - before >= swing and c > ema:
            side = SignalType.BUY
        elif now <= -entry and before - now >= swing and c < ema:
            side = SignalType.SELL
        else:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, c, float(_atr(df).iloc[-1]), float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, min(0.85, 0.55 + 0.3 * (abs(now) - entry)), stop_loss=sl, take_profit=tp,
                      metadata={"hl_top_net": now, "swing": now - before})


class COTInstitutionalStrategy(BaseStrategy):
    """
    Larry Williams' COT index on CME asset managers: where their net position
    sits within its range of the last `weeks` weeks (0..100). Enter when it
    crosses into the top (bottom) zone and price agrees with the EMA-50 trend.
    """
    SOURCE = ("CFTC Commitments of Traders (Traders in Financial Futures), CME Bitcoin/Ether "
              "futures — Larry Williams' COT index ('Trade Stocks and Commodities with the "
              "Insiders', 2005) applied to asset managers' net position")
    TUNABLE_PARAMS = {
        "weeks": ParamSpec(min=13, max=52, step=13),
        "index_high": ParamSpec(min=65, max=95, step=5),
        "index_low": ParamSpec(min=5, max=35, step=5),
        "atr_tp_mult": ParamSpec(min=2.0, max=8.0, step=0.5),
    }

    def __init__(self, params: dict = None):
        super().__init__("COT_Institucional", _defaults("COT_Institucional", params))

    @property
    def min_candles(self) -> int:
        return int(self.params["weeks"]) * 7 + 2

    max_hold_candles = 30

    @property
    def candle_interval(self) -> str:
        return self.params.get("candle_interval", "1d")

    def cot_index(self, net: pd.Series) -> pd.Series:
        window = int(self.params["weeks"]) * 7
        lo, hi = net.rolling(window).min(), net.rolling(window).max()
        return ((net - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        window = int(self.params["weeks"]) * 7
        if len(df) < self.min_candles or not _has(df, "cot_am_net", window + 1):
            return Signal(SignalType.HOLD, 0.0)
        idx = self.cot_index(df["cot_am_net"].iloc[-(window + 1):])
        now, prev = float(idx.iloc[-1]), float(idx.iloc[-2])
        hi, lo = float(self.params["index_high"]), float(self.params["index_low"])
        c, ema = float(df["close"].iloc[-1]), _ema(df["close"], 50)
        if now >= hi > prev and c > ema:
            side = SignalType.BUY
        elif now <= lo < prev and c < ema:
            side = SignalType.SELL
        else:
            return Signal(SignalType.HOLD, 0.0)
        sl, tp = _bracket(side, c, float(_atr(df).iloc[-1]), float(self.params["atr_sl_mult"]),
                          float(self.params["atr_tp_mult"]))
        return Signal(side, 0.65, stop_loss=sl, take_profit=tp,
                      metadata={"cot_index": now, "cot_am_net": float(df["cot_am_net"].iloc[-1])})
