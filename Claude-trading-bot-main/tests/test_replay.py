"""Part 4: accelerated replay drives the live engine on historical candles."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
import database as db


def test_intrabar_path_visits_levels_in_price_order():
    from run_replay_test import intrabar_path
    # Up candle: open -> low -> high -> close
    assert intrabar_path(100, 110, 95, 105, levels=[97, 108, 120, 50]) == \
        [100, 97, 95, 97, 108, 110, 108, 105]
    # Down candle: open -> high -> low -> close
    assert intrabar_path(100, 104, 90, 92, levels=[102, 91]) == [100, 102, 104, 102, 91, 90, 91, 92]


def _synthetic(days=900, seed=3):
    from utils.indicators import add_all_indicators
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp("2026-09-28", tz="UTC"), periods=days * 24, freq="1h")
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.004, len(idx))) * close
    h1 = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                       "low": np.minimum(open_, close) - spread, "close": close,
                       "volume": rng.uniform(50, 150, len(idx))}, index=idx)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    data = {"1h": h1}
    for iv, rule in (("4h", "4h"), ("1d", "1D")):
        data[iv] = add_all_indicators(h1.resample(rule).agg(agg).dropna())
    return data


def test_replay_runs_both_books_offline_and_keeps_baseline_frozen(tmp_path, monkeypatch):
    from run_replay_test import run_replay
    from strategies import ALL_STRATEGIES
    for name, value in (("LEARNING_PROPOSAL_DAYS", 30), ("LEARNING_VALIDATION_DAYS", 30),
                        ("LEARNING_MIN_VALIDATION_TRADES", 1), ("EVAL_WINDOWS", 2),
                        ("EVAL_WINDOW_DAYS", 20), ("EVAL_INTERVAL_HOURS", 24 * 7)):
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "replay.db"))   # restored after test
    data = _synthetic()
    end = data["1h"].index[-1] + pd.Timedelta(hours=1)
    start = end - pd.Timedelta(days=20)

    with patch("requests.get", side_effect=AssertionError("network")), \
         patch("requests.post", side_effect=AssertionError("network")):
        res = run_replay(data, start, end, db_path=str(tmp_path / "replay.db"))

    for book in ("learn", "lab", "baseline"):
        m = res["metrics"][book]
        assert m["equity_start"] == pytest.approx(config.INITIAL_CAPITAL)
        assert len(res["equity"][book]) == 20 * 24
    for b in res["baselines"]:
        assert b.frozen and b.params == type(b)().params
    assert len(db.get_learning_audit()) > 0          # the tuner ran on replay time
    assert all(r["ts"] < end.isoformat() for r in db.get_learning_audit())
    from strategies import ALL_STRATEGIES, CANDIDATE_STRATEGIES
    rated = db.get_all_strategy_status()             # the evaluator ran on replay time
    assert len(rated) == len(ALL_STRATEGIES) + len(CANDIDATE_STRATEGIES)
    assert all(e["ts"] < end.isoformat() for e in db.get_strategy_evaluations(limit=10**6))
    # the learning book never opened a position the evaluator did not allow
    blocked = [s for s in db.get_signals(book="observe", limit=10**6) if s["reason"].startswith("evaluator")]
    opened = [s for s in db.get_signals(book="observe", limit=10**6) if s["acted"]]
    assert blocked or not opened
    from run_replay_test import build_report
    report = build_report(res, "sintetico")
    assert "LAB" in report and "CALIFICACION FINAL" in report
