"""
Adaptive Tuner — safe, validated self-learning
───────────────────────────────────────────────
Every LEARNING_INTERVAL_HOURS, for each learning strategy that has enough new
data, propose ONE small change (± one step) to ONE declared tunable parameter:

  1. Proposal window  : [now - validation - proposal, now - validation)
     Backtest current params and both neighbours (value ± step). The best
     neighbour must beat the current value here, otherwise: rejected.
  2. Validation window: [now - validation, now) — data NOT used in step 1.
     Apply the candidate only if, out of sample, it has enough trades, a
     profit factor at least LEARNING_MIN_PF_IMPROVEMENT better, and a max
     drawdown no more than LEARNING_MAX_DD_WORSENING worse.
  3. Rollback         : LEARNING_ROLLBACK_WINDOW_HOURS after applying, compare
     the equity change of the learning strategy with its frozen baseline copy
     over the same period. If it lagged by more than
     LEARNING_ROLLBACK_TOLERANCE_PCT of its equity, restore the old value.

Hard limits: ParamSpec min/max, max LEARNING_MAX_CHANGES_PER_DAY changes per
strategy per day, and no new proposal while a change is still being evaluated.
Every proposal (applied / rejected / rollback) is written to learning_audit.

Purely numerical: no LLM and no API key involved.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

import pandas as pd

import config
import database as db
from backtester import Backtester
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    trades: int
    profit_factor: float
    max_drawdown: float
    total_pnl: float

    @classmethod
    def from_backtest(cls, result) -> "Metrics":
        return cls(trades=result.total_trades, profit_factor=float(result.profit_factor),
                   max_drawdown=float(result.max_drawdown), total_pnl=float(result.total_pnl))

    @property
    def pf(self) -> float:
        return min(self.profit_factor, config.LEARNING_PF_CAP)

    def as_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(self).items()}


def backtest_window(strategy: BaseStrategy, df: pd.DataFrame, window: str) -> Metrics:
    """Default evaluator: the project's Backtester on df (warm-up rows included)."""
    capital = config.INITIAL_CAPITAL / max(config.MAX_STRATEGIES, 1)
    return Metrics.from_backtest(Backtester(strategy, df, initial_capital=capital).run())


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat()


def _parse(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


class AdaptiveTuner:

    def __init__(self, learners: Dict[str, BaseStrategy],
                 history_fn: Callable[[str, int, datetime], pd.DataFrame],
                 equity_fn: Callable[[str, str], float],
                 clock: Callable[[], datetime],
                 learner_book: str = "main",
                 baseline_book: str = "baseline",
                 evaluate_fn: Callable[[BaseStrategy, pd.DataFrame, str], Metrics] = backtest_window):
        """
        learners     : name -> learning strategy instance (params are modified in place)
        history_fn   : (interval, days, end) -> closed OHLCV candles up to `end`
        equity_fn    : (book, strategy_name) -> current equity incl. unrealized PnL
        clock        : current UTC time (the replay injects historical time)
        """
        self.learners = learners
        self.history_fn = history_fn
        self.equity_fn = equity_fn
        self.clock = clock
        self.learner_book = learner_book
        self.baseline_book = baseline_book
        self.evaluate_fn = evaluate_fn
        self._history_cache: Dict[str, pd.DataFrame] = {}

    # ─── Persistence ─────────────────────────────────────────────────────────

    @staticmethod
    def restore_learned_params(learners: Dict[str, BaseStrategy]):
        """Re-apply learned values saved by a previous run (declared params only)."""
        for name, strat in learners.items():
            saved = db.get_meta(f"learned_params:{name}")
            if saved:
                strat.restore_tunables(json.loads(saved))
                logger.info(f"[Learning] {name}: restored learned params {strat.tunable_values()}")

    @staticmethod
    def _save_params(strat: BaseStrategy):
        db.set_meta(f"learned_params:{strat.name}", json.dumps(strat.tunable_values()))

    # ─── Main entry point ────────────────────────────────────────────────────

    def run_cycle_if_due(self):
        now = self.clock()
        self._evaluate_pending(now)

        last = db.get_meta("learning:last_cycle")
        if last and now - _parse(last) < timedelta(hours=config.LEARNING_INTERVAL_HOURS):
            return
        db.set_meta("learning:last_cycle", _iso(now))

        self._history_cache.clear()
        for strat in self.learners.values():
            if strat.is_active and not strat.frozen and strat.TUNABLE_PARAMS:
                try:
                    self._propose(strat, now)
                except Exception as e:
                    logger.error(f"[Learning] {strat.name}: proposal failed: {e}", exc_info=True)

    # ─── Proposal + walk-forward validation ──────────────────────────────────

    def _propose(self, strat: BaseStrategy, now: datetime):
        name = strat.name
        if any(p["strategy_name"] == name for p in db.get_pending_learning_changes()):
            return   # previous change still in its rollback window

        day_start = _iso(now.replace(hour=0, minute=0, second=0, microsecond=0))
        today = [r for r in db.get_learning_audit(name, decision="applied") if r["ts"] >= day_start]
        if len(today) >= config.LEARNING_MAX_CHANGES_PER_DAY:
            return

        audit = db.get_learning_audit(name, limit=1)
        if audit:
            last_ts = audit[0]["ts"]
            new_trades = db.count_trades_since(name, last_ts, self.learner_book)
            new_days = (now - _parse(last_ts)).total_seconds() / 86400
            if new_trades < config.LEARNING_MIN_NEW_TRADES and new_days < config.LEARNING_MIN_NEW_DAYS:
                return

        param = self._next_param(strat)
        spec = strat.TUNABLE_PARAMS[param]
        current = strat.params[param]
        candidates = sorted({spec.clamp(current + spec.step), spec.clamp(current - spec.step)} - {current})

        val_start = now - timedelta(days=config.LEARNING_VALIDATION_DAYS)
        prop_start = val_start - timedelta(days=config.LEARNING_PROPOSAL_DAYS)
        df = self._history(strat.candle_interval, now)
        prop_win = self._window(df, strat, prop_start, val_start)
        val_win = self._window(df, strat, val_start, now)
        base = dict(ts=_iso(now), strategy_name=name, param=param, old_value=current,
                    book=self.learner_book)
        if prop_win is None or val_win is None or not candidates:
            db.record_learning_audit(decision="rejected", reason="not enough history", **base)
            return

        def evaluate(value, win, label):
            clone = strat.clone({param: value})
            i0, i1 = win   # each clone gets exactly its own warm-up rows
            return self.evaluate_fn(clone, df.iloc[i0 - clone.min_candles:i1], label)

        # 1. Proposal window: pick the best neighbour, must beat the current value.
        cur_prop = evaluate(current, prop_win, "proposal")
        best_val, best_m = None, None
        for value in candidates:
            m = evaluate(value, prop_win, "proposal")
            if m.trades > 0 and m.pf > cur_prop.pf and \
                    m.max_drawdown <= cur_prop.max_drawdown + config.LEARNING_MAX_DD_WORSENING:
                if best_m is None or m.pf > best_m.pf:
                    best_val, best_m = value, m
        if best_val is None:
            db.record_learning_audit(
                decision="rejected", reason="no neighbour improves the proposal window",
                metrics_before=cur_prop.as_dict(), **base)
            logger.info(f"[Learning] {name}.{param}={current}: no improvement proposed")
            return

        # 2. Validation window (never used to choose the candidate).
        cur_val = evaluate(current, val_win, "validation")
        new_val = evaluate(best_val, val_win, "validation")
        ok, reason = self._accepts(cur_val, new_val)
        base.update(new_value=best_val, metrics_before=cur_val.as_dict(),
                    metrics_after=new_val.as_dict())
        if not ok:
            db.record_learning_audit(decision="rejected", reason=reason, **base)
            logger.info(f"[Learning] {name}.{param} {current}->{best_val} rejected: {reason}")
            return

        applied = strat.set_tunable_param(param, best_val)
        self._save_params(strat)
        db.record_learning_audit(
            decision="applied", reason=reason,
            evaluate_after=_iso(now + timedelta(hours=config.LEARNING_ROLLBACK_WINDOW_HOURS)),
            learner_equity=self.equity_fn(self.learner_book, name),
            baseline_equity=self.equity_fn(self.baseline_book, name),
            **{**base, "new_value": applied})
        logger.info(f"[Learning] {name}.{param} {current}->{applied} APPLIED: {reason}")

    @staticmethod
    def _accepts(before: Metrics, after: Metrics):
        if after.trades < config.LEARNING_MIN_VALIDATION_TRADES:
            return False, (f"validation: only {after.trades} trades "
                           f"(< {config.LEARNING_MIN_VALIDATION_TRADES})")
        if after.pf < before.pf * (1 + config.LEARNING_MIN_PF_IMPROVEMENT) or after.pf <= 1.0:
            return False, (f"validation: profit factor {after.pf:.2f} vs {before.pf:.2f} "
                           f"(needs +{config.LEARNING_MIN_PF_IMPROVEMENT:.0%} and > 1)")
        if after.max_drawdown > before.max_drawdown + config.LEARNING_MAX_DD_WORSENING:
            return False, (f"validation: max drawdown {after.max_drawdown:.1%} vs "
                           f"{before.max_drawdown:.1%}")
        return True, (f"validation PF {before.pf:.2f}->{after.pf:.2f}, "
                      f"maxDD {before.max_drawdown:.1%}->{after.max_drawdown:.1%}, "
                      f"{after.trades} trades")

    # ─── Rollback ────────────────────────────────────────────────────────────

    def _evaluate_pending(self, now: datetime):
        for change in db.get_pending_learning_changes():
            if not change["evaluate_after"] or now < _parse(change["evaluate_after"]):
                continue
            name = change["strategy_name"]
            strat = self.learners.get(name)
            if strat is None:
                continue
            learner_delta = self.equity_fn(self.learner_book, name) - (change["learner_equity"] or 0)
            baseline_delta = self.equity_fn(self.baseline_book, name) - (change["baseline_equity"] or 0)
            tolerance = config.LEARNING_ROLLBACK_TOLERANCE_PCT * max(change["learner_equity"] or 0, 1)
            db.mark_learning_change_evaluated(change["id"])
            gap = learner_delta - baseline_delta
            if gap < -tolerance and strat.params.get(change["param"]) == change["new_value"]:
                old = strat.set_tunable_param(change["param"], change["old_value"])
                self._save_params(strat)
                db.record_learning_audit(
                    ts=_iso(now), strategy_name=name, param=change["param"],
                    old_value=change["new_value"], new_value=old, decision="rollback",
                    reason=(f"after {config.LEARNING_ROLLBACK_WINDOW_HOURS:.0f}h learner "
                            f"{learner_delta:+.2f} vs baseline {baseline_delta:+.2f} "
                            f"(tolerance {tolerance:.2f})"),
                    book=self.learner_book,
                    metrics_before={"learner_delta": round(learner_delta, 4),
                                    "baseline_delta": round(baseline_delta, 4)})
                logger.warning(f"[Learning] {name}.{change['param']} ROLLED BACK to {old}")
            else:
                logger.info(f"[Learning] {name}.{change['param']}={change['new_value']} kept "
                            f"(learner {learner_delta:+.2f} vs baseline {baseline_delta:+.2f})")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _next_param(self, strat: BaseStrategy) -> str:
        names = list(strat.TUNABLE_PARAMS)
        key = f"learning:next_param:{strat.name}"
        idx = int(db.get_meta(key) or 0) % len(names)
        db.set_meta(key, str(idx + 1))
        return names[idx]

    def _history(self, interval: str, now: datetime) -> pd.DataFrame:
        if interval not in self._history_cache:
            days = (config.LEARNING_PROPOSAL_DAYS + config.LEARNING_VALIDATION_DAYS
                    + config.LEARNING_WARMUP_DAYS)
            self._history_cache[interval] = self.history_fn(interval, days, now)
        return self._history_cache[interval]

    @staticmethod
    def _window(df: pd.DataFrame, strat: BaseStrategy, start: datetime,
                end: datetime) -> Optional[tuple]:
        """
        Row bounds (i0, i1) of the candles in [start, end), or None if there is
        not enough history before `start` to warm up any allowed param value.
        """
        if df is None or df.empty:
            return None
        warm = max(strat.min_candles,
                   *(strat.clone({p: s.max}).min_candles for p, s in strat.TUNABLE_PARAMS.items()))
        i0, i1 = df.index.searchsorted(start), df.index.searchsorted(end)
        if i0 < warm or i1 - i0 < 2:
            return None
        return i0, i1
