"""
Strategy Evaluator — is each strategy viable, when and how?
───────────────────────────────────────────────────────────
Every EVAL_INTERVAL_HOURS, for every registered and candidate strategy that has
not been discarded:

  • Effectiveness: backtest on EVAL_WINDOWS consecutive, non-overlapping windows
    of EVAL_WINDOW_DAYS (walk-forward over ~2 years), net of fees.
  • When: each trade is tagged with the market regime at entry
    (TRENDING_UP / TRENDING_DOWN / RANGING from ADX + EMA-50/200).
  • How: results split by side (LONG / SHORT); the tuned params are used.
  • Live evidence: its paper trades in the 'lab' book (every non-discarded
    strategy trades there without filters).

Rating (classify):
  VIABLE      enough trades, PF >= 1.2, profitable in >= 3 of 4 windows,
              worst drawdown <= 15%, and not losing in the MOST RECENT window
                                                     → trades in any regime
  CONDICIONAL not viable overall, but PF >= 1.3 with >= 10 trades in some
              regime, and not losing in the most recent window
                                                     → trades only in those regimes
  EN_PRUEBA   not enough / mixed evidence, or it used to work but lost in the
              last window                            → does not trade (lab only)
  DESCARTADA  >= 40 trades, PF < 0.9, profitable in <= 1 window and no regime
              where it works; or >= 60 trades and PF < 1 (no edge after fees)
              and no regime where it works; or clearly losing live
                                                     → value 0, never
                                                       re-evaluated nor traded again
Window counts are "of 4" and scale with EVAL_WINDOWS (3 of 4 = 6 of 8).
A side (LONG/SHORT) with >= 10 trades and PF < 0.9 is blocked.

Why the recent-window rule and the no-edge discard: in the 5-coin replays the
best-looking combinations on past data did not carry forward (selection bias
over 125 combinations); many 'mixed' strategies with dozens of trades and
PF < 1 stayed EN_PRUEBA forever.

Purely numerical: no LLM involved.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import database as db
from backtester import Backtester
from strategies.base_strategy import BaseStrategy, SignalType
from utils import symbol_of

logger = logging.getLogger(__name__)

REGIMES = ("TRENDING_UP", "TRENDING_DOWN", "RANGING")
REGIME_ES = {"TRENDING_UP": "tendencia alcista", "TRENDING_DOWN": "tendencia bajista",
             "RANGING": "lateral"}
STATUS_ORDER = ("VIABLE", "CONDICIONAL", "EN_PRUEBA", "DESCARTADA")
ADX_TREND = 22.0


def regime_series(df: pd.DataFrame) -> pd.Series:
    """Market regime per candle from ADX and the EMA-50 / EMA-200 alignment."""
    from utils.indicators import _adx, _ema
    close = df["close"]
    ema50 = df["ema_50"] if "ema_50" in df else _ema(close, 50)
    ema200 = df["ema_200"] if "ema_200" in df else _ema(close, 200)
    adx = df["adx"] if "adx" in df else _adx(df["high"], df["low"], close, 14)[0]
    trending = adx >= ADX_TREND
    up = trending & (close > ema200) & (ema50 > ema200)
    down = trending & (close < ema200) & (ema50 < ema200)
    return pd.Series(np.where(up, "TRENDING_UP", np.where(down, "TRENDING_DOWN", "RANGING")),
                     index=df.index)


def _pf(pnls) -> float:
    win = sum(p for p in pnls if p > 0)
    loss = -sum(p for p in pnls if p <= 0)
    if loss > 0:
        return min(win / loss, config.LEARNING_PF_CAP)
    return config.LEARNING_PF_CAP if win > 0 else 0.0


def _stats(pnls) -> dict:
    return {"trades": len(pnls), "profit_factor": round(_pf(pnls), 3),
            "pnl": round(float(sum(pnls)), 2),
            "win_rate": round(sum(p > 0 for p in pnls) / len(pnls), 3) if pnls else 0.0}


def default_backtest(strategy: BaseStrategy, df: pd.DataFrame):
    """[(entry_time, side, net_pnl)], max_drawdown — with the project's Backtester."""
    capital = config.INITIAL_CAPITAL / max(config.MAX_STRATEGIES, 1)
    r = Backtester(strategy, df, initial_capital=capital).run()
    return [(t.entry_time, t.side, t.pnl - t.fees) for t in r.trades], r.max_drawdown


def _not_viable_because(m: dict) -> str:
    why = []
    if m["trades"] < config.VIABLE_MIN_TRADES:
        why.append(f"{m['trades']} trades < {config.VIABLE_MIN_TRADES}")
    if m["profit_factor"] < config.VIABLE_MIN_PF:
        why.append(f"PF {m['profit_factor']:.2f} < {config.VIABLE_MIN_PF}")
    if m["profitable_windows"] < config.VIABLE_MIN_WINDOWS:
        why.append(f"rentable en {m['profitable_windows']}/{m['windows']} ventanas")
    if m["worst_drawdown"] > config.VIABLE_MAX_DRAWDOWN:
        why.append(f"drawdown {m['worst_drawdown']:.1%}")
    return ", ".join(why) or "no cumple los minimos"


def classify(m: dict) -> dict:
    """Pure rating from aggregated metrics (see module docstring)."""
    n, pf, pw, windows = m["trades"], m["profit_factor"], m["profitable_windows"], m["windows"]
    # thresholds are "of 4 windows" and scale with the number of windows
    need_windows = math.ceil(config.VIABLE_MIN_WINDOWS * max(windows, 1) / 4)
    max_discard_windows = math.floor(config.DISCARD_MAX_WINDOWS * max(windows, 1) / 4)
    recent = (m.get("window_pnl") or [None])[-1]
    recent_loss = recent is not None and recent < 0
    good_regimes = [r for r, s in m["by_regime"].items()
                    if s["trades"] >= config.COND_MIN_REGIME_TRADES
                    and s["profit_factor"] >= config.COND_MIN_REGIME_PF]
    sides = [s for s in ("LONG", "SHORT")
             if not (m["by_side"].get(s, {}).get("trades", 0) >= config.SIDE_BLOCK_MIN_TRADES
                     and m["by_side"][s]["profit_factor"] < config.SIDE_BLOCK_MAX_PF)]
    live = m.get("live") or {}
    sufficiency = min(1.0, n / config.VIABLE_MIN_TRADES)
    quality = min(max((min(pf, 3.0) - 0.8) / 0.8, 0.0), 1.0)
    score = round(sufficiency * (0.5 * pw / max(windows, 1) + 0.5 * quality), 2)
    out = dict(allowed_regimes=[], allowed_sides=sides, score=score)

    if (n >= config.DISCARD_MIN_TRADES and pf < config.DISCARD_MAX_PF
            and pw <= max_discard_windows and not good_regimes):
        return {**out, "status": "DESCARTADA", "score": 0.0, "allowed_sides": [],
                "reason": f"pierde de forma consistente: {n} trades, PF {pf:.2f}, rentable en "
                          f"{pw}/{windows} ventanas y en ningun tipo de mercado"}
    if (live.get("trades", 0) >= config.DISCARD_LIVE_MIN_TRADES
            and live.get("profit_factor", 1) < config.DISCARD_LIVE_MAX_PF and pf < 1.0):
        return {**out, "status": "DESCARTADA", "score": 0.0, "allowed_sides": [],
                "reason": f"pierde en vivo ({live['trades']} trades, PF {live['profit_factor']:.2f}) "
                          f"y en historico (PF {pf:.2f})"}
    if n >= config.EVAL_NO_EDGE_MIN_TRADES and pf < 1.0 and not good_regimes:
        return {**out, "status": "DESCARTADA", "score": 0.0, "allowed_sides": [],
                "reason": f"sin ventaja tras {n} trades: PF {pf:.2f} < 1 despues de comisiones, "
                          f"rentable en {pw}/{windows} ventanas y en ningun tipo de mercado"}
    if not sides:
        return {**out, "status": "EN_PRUEBA", "reason": "ambas direcciones pierden"}
    viable = (n >= config.VIABLE_MIN_TRADES and pf >= config.VIABLE_MIN_PF
              and pw >= need_windows and m["worst_drawdown"] <= config.VIABLE_MAX_DRAWDOWN)
    if (viable or good_regimes) and recent_loss:
        return {**out, "status": "EN_PRUEBA",
                "reason": f"funcionaba antes (PF {pf:.2f}, rentable en {pw}/{windows} ventanas), pero "
                          f"pierde en la ventana mas reciente ({recent:+.2f})"}
    if viable:
        return {**out, "status": "VIABLE", "allowed_regimes": list(REGIMES),
                "reason": f"{n} trades, PF {pf:.2f}, rentable en {pw}/{windows} ventanas, "
                          f"peor drawdown {m['worst_drawdown']:.1%}"}
    if good_regimes:
        where = ", ".join(REGIME_ES[r] for r in good_regimes)
        return {**out, "status": "CONDICIONAL", "allowed_regimes": good_regimes,
                "reason": f"en conjunto no es viable ({_not_viable_because(m)}), "
                          f"pero funciona en: {where}"}
    if n < config.VIABLE_MIN_TRADES:
        why = f"pocos trades ({n} < {config.VIABLE_MIN_TRADES}) para decidir"
    else:
        why = f"resultados mixtos: PF {pf:.2f}, rentable en {pw}/{windows} ventanas"
    return {**out, "status": "EN_PRUEBA", "reason": why}


class StrategyEvaluator:

    def __init__(self, strategies: Dict[str, BaseStrategy],
                 history_fn: Callable[[str, int, datetime, str], pd.DataFrame],
                 clock: Callable[[], datetime],
                 live_book: str = "lab",
                 backtest_fn: Callable = default_backtest):
        self.strategies = strategies
        self.history_fn = history_fn
        self.clock = clock
        self.live_book = live_book
        self.backtest_fn = backtest_fn
        self._cache: Dict[str, pd.DataFrame] = {}

    # ─── Cycle ───────────────────────────────────────────────────────────────

    def run_cycle_if_due(self):
        now = self.clock()
        last = db.get_meta("evaluator:last_cycle")
        if last and now - datetime.fromisoformat(last) < timedelta(hours=config.EVAL_INTERVAL_HOURS):
            return
        db.set_meta("evaluator:last_cycle", now.isoformat())
        self._cache.clear()
        for strat in self.strategies.values():
            st = db.get_strategy_status(strat.name)
            if st and st["status"] == "DESCARTADA":
                continue            # value 0 for good: never evaluated nor traded again
            try:
                self.evaluate(strat, now)
            except Exception as e:
                logger.error(f"[Evaluator] {strat.name}: {e}", exc_info=True)

    def evaluate(self, strat: BaseStrategy, now: datetime) -> dict:
        interval, symbol = strat.candle_interval, symbol_of(strat)
        if (symbol, interval) not in self._cache:     # each symbol is rated on its own candles
            days = config.EVAL_WINDOWS * config.EVAL_WINDOW_DAYS + config.EVAL_WARMUP_DAYS
            self._cache[(symbol, interval)] = self.history_fn(interval, days, now, symbol)
        df = self._cache[(symbol, interval)]
        regimes = regime_series(df) if len(df) else pd.Series(dtype=object)

        all_trades, window_pnls, worst_dd = [], [], 0.0
        k, span = config.EVAL_WINDOWS, timedelta(days=config.EVAL_WINDOW_DAYS)
        for w in range(k):
            start = now - span * (k - w)
            end = start + span
            i0, i1 = int((df.index < start).sum()), int((df.index < end).sum())
            clone = strat.clone()
            if i0 < clone.min_candles or i1 - i0 < 2:
                continue
            trades, dd = self.backtest_fn(clone, df.iloc[i0 - clone.min_candles:i1])
            window_pnls.append(sum(p for _, _, p in trades))
            worst_dd = max(worst_dd, float(dd))
            all_trades.extend(trades)

        pnls = [p for _, _, p in all_trades]
        by_regime: Dict[str, list] = {}
        by_side: Dict[str, list] = {}
        for t, side, p in all_trades:
            reg = regimes.asof(t) if len(regimes) else "RANGING"
            by_regime.setdefault(str(reg), []).append(p)
            by_side.setdefault(side, []).append(p)
        live = _stats([float(t["pnl"]) for t in db.get_trades(strat.name, limit=10**6,
                                                               book=self.live_book)])
        capital = config.INITIAL_CAPITAL / max(config.MAX_STRATEGIES, 1)
        metrics = {
            **_stats(pnls),
            "windows": len(window_pnls),
            "profitable_windows": sum(p > 0 for p in window_pnls),
            "window_pnl": [round(p, 2) for p in window_pnls],
            "worst_drawdown": round(worst_dd, 4),
            "expectancy_pct": round(float(np.mean(pnls)) / capital, 5) if pnls else 0.0,
            "by_regime": {r: _stats(v) for r, v in by_regime.items()},
            "by_side": {s: _stats(v) for s, v in by_side.items()},
            "live": live,
            "params": strat.tunable_values(),
            "interval": interval,
        }
        rating = classify(metrics)
        ts = now.isoformat()
        db.upsert_strategy_status(strat.name, rating["status"], rating["score"],
                                  rating["allowed_regimes"], rating["allowed_sides"],
                                  rating["reason"], metrics, ts=ts)
        db.record_strategy_evaluation(ts, strat.name, rating["status"], rating["score"],
                                      rating["reason"], metrics)
        logger.info(f"[Evaluator] {strat.name}: {rating['status']} score={rating['score']:.2f} "
                    f"— {rating['reason']}")
        return rating

    # ─── Gate used by the trading loop ───────────────────────────────────────

    def can_trade(self, strat: BaseStrategy, side: SignalType, df: pd.DataFrame,
                  allowed_statuses=("VIABLE", "CONDICIONAL")) -> Tuple[bool, str]:
        """allowed_statuses comes from the aggressiveness (risk_engine.profile);
        a DESCARTADA strategy never trades."""
        st = db.get_strategy_status(strat.name)
        if st is None:
            return False, "evaluator: sin evaluar todavia"
        if st["status"] == "DESCARTADA":
            return False, "evaluator: DESCARTADA (valor 0)"
        if st["status"] not in allowed_statuses:
            return False, f"evaluator: {st['status']} (no permitido con esta agresividad)"
        side_name = "LONG" if side == SignalType.BUY else "SHORT"
        if side_name not in st["allowed_sides"]:
            return False, f"evaluator: {side_name} bloqueado"
        if st["status"] == "CONDICIONAL":
            regime = str(regime_series(df).iloc[-1])
            if regime not in st["allowed_regimes"]:
                return False, f"evaluator: CONDICIONAL, mercado {regime} no permitido"
        return True, ""

    @staticmethod
    def is_discarded(name: str) -> bool:
        st = db.get_strategy_status(name)
        return bool(st and st["status"] == "DESCARTADA")
