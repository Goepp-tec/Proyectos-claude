"""Part 5: 7-day test runner — survives restarts, hourly snapshots, daily CSV, final report."""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
import database as db
from portfolio_manager import PortfolioManager
from tests.test_portfolio_risk import _buy, _strategy

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def run(temp_db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 10_000.0)
    from scripts.run_7day_test import TestRun
    return TestRun(name="t", duration_hours=168, snapshot_minutes=60, out_dir=str(tmp_path))


def _bot():
    learners = [_strategy("A"), _strategy("B")]
    baselines = [_strategy("A"), _strategy("B")]
    return SimpleNamespace(
        book="observe", strategies=learners, baselines=baselines, _current_price=50_000.0,
        portfolio=PortfolioManager(Mock(), learners, book="observe", simulate_fills=True),
        baseline_portfolio=PortfolioManager(Mock(), baselines, book="baseline", simulate_fills=True),
    )


def test_start_and_resume_keep_the_schedule_and_log_interruptions(run):
    assert run.start_or_resume(T0) == "started"
    assert run.end == T0 + timedelta(days=7)
    run.heartbeat(T0 + timedelta(minutes=10))

    # Container / server restart two hours later
    from scripts.run_7day_test import TestRun
    again = TestRun(name="t", duration_hours=168, snapshot_minutes=60, out_dir=run.out_dir)
    assert again.start_or_resume(T0 + timedelta(hours=2)) == "resumed"
    assert again.start == T0 and again.end == T0 + timedelta(days=7)
    gaps = again.interruptions()
    assert len(gaps) == 1
    assert gaps[0]["down_from"] == (T0 + timedelta(minutes=10)).isoformat()
    assert gaps[0]["down_to"] == (T0 + timedelta(hours=2)).isoformat()


def test_resume_does_not_reset_equity_or_history(run):
    bot = _bot()
    run.start_or_resume(T0)
    bot.portfolio.process_signal(bot.portfolio.strategies["A"], _buy(50_000.0), 50_000.0, 0.6)
    from scripts.run_7day_test import TestRun
    TestRun(name="t", duration_hours=168, snapshot_minutes=60,
            out_dir=run.out_dir).start_or_resume(T0 + timedelta(hours=1))
    assert len(db.get_open_positions(book="observe")) == 1


def test_hourly_snapshots_per_book_and_strategy(run):
    bot = _bot()
    run.start_or_resume(T0)
    assert run.snapshot_due(T0)
    run.snapshot(T0, bot)
    assert not run.snapshot_due(T0 + timedelta(minutes=30))
    assert run.snapshot_due(T0 + timedelta(minutes=60))

    rows = run.snapshots()
    names = {(r["book"], r["strategy_name"]) for r in rows}
    assert names == {("observe", "A"), ("observe", "B"), ("observe", "TOTAL"),
                     ("baseline", "A"), ("baseline", "B"), ("baseline", "TOTAL")}
    total = next(r for r in rows if r["book"] == "observe" and r["strategy_name"] == "TOTAL")
    assert total["equity"] == pytest.approx(10_000.0)
    assert total["drawdown_pct"] == 0.0


def test_snapshots_include_the_lab_book_when_present(run):
    bot = _bot()
    lab = [_strategy("A"), _strategy("B")]
    bot.lab, bot.lab_portfolio = lab, PortfolioManager(Mock(), lab, book="lab", simulate_fills=True)
    run.start_or_resume(T0)
    run.snapshot(T0, bot)
    books = {r["book"] for r in run.snapshots()}
    assert books == {"observe", "baseline", "lab"}
    text = run.report(T0 + timedelta(hours=1), bot.book)
    assert "LAB" in text and "mismo capital" in text


def test_budget_book_metrics_use_the_budget_and_ignore_budget_changes(run):
    bot = _bot()
    bot.portfolio = PortfolioManager(Mock(), bot.strategies, book="observe", simulate_fills=True,
                                     capital_base=100.0)
    run.start_or_resume(T0)
    run.snapshot(T0, bot)
    bot.portfolio.set_capital_base(250.0)          # budget raised from the dashboard
    run.snapshot(T0 + timedelta(hours=1), bot)
    m = run.book_metrics("observe")
    assert m["equity_start"] == pytest.approx(100.0)
    assert m["total_return"] == pytest.approx(0.0, abs=1e-9)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)     # not a 'drop' from 250 to 100
    total = [r for r in run.snapshots() if r["book"] == "observe" and r["strategy_name"] == "TOTAL"]
    assert [r["capital_base"] for r in total] == [pytest.approx(100.0), pytest.approx(250.0)]


def test_finish_early_without_a_running_bot(run):
    bot = _bot()
    run.start_or_resume(T0)
    run.snapshot(T0, bot)
    db.set_meta("trading_mode", "OBSERVE")
    path = run.finish_early(T0 + timedelta(days=2), note="cerrada a mano para arrancar la prueba 2")
    text = open(path, encoding="utf-8").read()
    assert "cerrada a mano" in text and "APRENDE" in text
    assert run.finished


def test_daily_csv_export(run):
    bot = _bot()
    run.start_or_resume(T0)
    run.snapshot(T0, bot)
    assert run.export_daily(T0 + timedelta(hours=3)) == []            # day 1 not over
    files = run.export_daily(T0 + timedelta(days=1, minutes=1))
    assert files and all(os.path.exists(f) for f in files)
    assert run.export_daily(T0 + timedelta(days=1, minutes=5)) == []  # once per day


def test_verdict_labels():
    from scripts.run_7day_test import verdict
    good = dict(equity_start=10_000, equity_end=10_300, max_drawdown=0.03, trades=25,
                profit_factor=1.5)
    base = dict(equity_end=10_100)
    assert verdict(good, base)[0].startswith("CUMPLE")
    assert verdict({**good, "trades": 5}, base)[0].startswith("NO CONCLUYENTE")
    assert verdict({**good, "equity_end": 9_800, "profit_factor": 0.8}, base)[0] == "NO RENTABLE"
    assert verdict(good, {"equity_end": 10_500})[0].startswith("NO CONCLUYENTE")


def test_finish_writes_report_and_is_final(run):
    bot = _bot()
    run.start_or_resume(T0)
    run.snapshot(T0, bot)
    end = run.end
    assert not run.is_over(end - timedelta(minutes=1)) and run.is_over(end)
    db.upsert_strategy_status("A", "DESCARTADA", 0.0, [], [], "pierde de forma consistente", {})
    db.record_strategy_evaluation(T0.isoformat(), "A", "DESCARTADA", 0.0, "pierde de forma consistente", {})
    path = run.finish(end, bot)
    text = open(path, encoding="utf-8").read()
    assert os.path.basename(path) == "REPORTE_7DIAS.txt"
    for s in ("REPORTE", "Interrupciones", "APRENDE", "BASELINE", "Veredicto",
              "Evaluador de estrategias", "DESCARTADA", "pierde de forma consistente"):
        assert s in text
    from scripts.run_7day_test import TestRun
    assert TestRun(name="t", duration_hours=168, snapshot_minutes=60,
                   out_dir=run.out_dir).start_or_resume(end + timedelta(hours=1)) == "finished"


def test_snapshot_values_each_symbol_at_its_own_price(run):
    """With several coins a position must not be valued at BTC's price."""
    eth = Mock()
    eth.name, eth.symbol, eth.is_active = "A@ETH", "ETHUSDT", True
    pm = PortfolioManager(Mock(), [eth], book="observe", simulate_fills=True)
    prices = {"BTCUSDT": 50_000.0, "ETHUSDT": 2_000.0}
    bot = SimpleNamespace(book="observe", strategies=[eth], baselines=[], _current_price=50_000.0,
                          prices=lambda: dict(prices), portfolio=pm,
                          baseline_portfolio=PortfolioManager(Mock(), [], book="baseline",
                                                              simulate_fills=True))
    run.start_or_resume(T0)
    pm.process_signal(eth, _buy(2_000.0), 2_000.0, 0.6)
    pos = db.get_open_positions("A@ETH", book="observe")[0]
    prices["ETHUSDT"] = 2_100.0
    run.snapshot(T0, bot)
    row = next(r for r in run.snapshots() if r["strategy_name"] == "A@ETH")
    assert row["unrealized_pnl"] == pytest.approx((2_100.0 - pos["entry_price"]) * pos["quantity"])
