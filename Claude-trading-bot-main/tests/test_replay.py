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
        res = run_replay(data, start, end, db_path=str(tmp_path / "replay.db"),
                         funds=1_000, budget=100, aggressiveness=8, collect_market=False)

    assert res["metrics"]["learn"]["equity_start"] == pytest.approx(100)       # the budget
    for book in ("lab", "baseline"):
        assert res["metrics"][book]["equity_start"] == pytest.approx(1_000)    # the funds
    for book in ("learn", "lab", "baseline"):
        assert len(res["equity"][book]) == 20 * 24
    # the learning book never had more than the budget in open positions
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "replay.db"))
    assert all(float(r[0]) <= 100 * 1.001 for r in conn.execute(
        "SELECT entry_price*quantity FROM positions WHERE book='observe'"))
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
    # every learning-book trade carries the checklist of confirmations it passed
    import json
    learn_trades = db.get_trades(limit=10**6, book="observe")
    assert all("confirmaciones" in (t["entry_features"] if isinstance(t["entry_features"], dict)
                                     else json.loads(t["entry_features"] or "{}")) for t in learn_trades)
    from run_replay_test import build_report
    report = build_report(res, "sintetico")
    assert "LAB" in report and "CALIFICACION FINAL" in report and "CONFIRMACIONES" in report
    # different capital bases (budget vs funds): the verdict compares returns, not dollars
    verdict = next(l for l in report.splitlines() if l.startswith("Aprende vs baseline"))
    assert "USD" not in verdict and "pts" in verdict


def test_replay_trades_several_symbols_each_on_its_own_candles(tmp_path, monkeypatch):
    from run_replay_test import build_report, run_replay
    from strategies import ALL_STRATEGIES, CANDIDATE_STRATEGIES
    for name, value in (("LEARNING_PROPOSAL_DAYS", 30), ("LEARNING_VALIDATION_DAYS", 30),
                        ("EVAL_WINDOWS", 2), ("EVAL_WINDOW_DAYS", 20), ("EVAL_INTERVAL_HOURS", 24 * 7)):
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "replay2.db"))
    btc, eth = _synthetic(seed=3), _synthetic(seed=8)
    for iv in eth:                                   # ETH trades around 2,000, BTC around 30,000
        eth[iv] = eth[iv].assign(**{c: eth[iv][c] / 15 for c in ("open", "high", "low", "close")})
    end = btc["1h"].index[-1] + pd.Timedelta(hours=1)
    start = end - pd.Timedelta(days=20)

    with patch("requests.get", side_effect=AssertionError("network")), \
         patch("requests.post", side_effect=AssertionError("network")):
        res = run_replay({"BTCUSDT": btc, "ETHUSDT": eth}, start, end,
                         db_path=str(tmp_path / "replay2.db"), funds=1_000, budget=100,
                         aggressiveness=8, collect_market=False)

    for book in ("lab", "baseline"):             # 1,000 per symbol: same share per strategy
        assert res["metrics"][book]["equity_start"] == pytest.approx(2_000)
    trades = db.get_trades(limit=10**6, book="lab")
    assert {t["symbol"] for t in trades} == {"BTCUSDT", "ETHUSDT"}
    for t in trades:                             # every fill at its own symbol's price level
        assert (t["entry_price"] > 5_000) == (t["symbol"] == "BTCUSDT")
    rated = db.get_all_strategy_status()
    assert len(rated) == 2 * (len(ALL_STRATEGIES) + len(CANDIDATE_STRATEGIES))
    assert any(r["strategy_name"].endswith("@ETH") for r in rated)
    assert set(res["buy_hold"]) == {"BTCUSDT", "ETHUSDT"}
    assert "RESULTADO POR CRIPTO" in build_report(res, "sintetico")
