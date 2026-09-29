"""Phase 4: the bot writes a daily report (UTC day) — results vs buy & hold, per
coin, strategies added / removed by the evaluator, learning, risk, market."""

import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock

import pandas as pd
import pytest

import config
import database as db
from portfolio_manager import PortfolioManager
from strategies.base_strategy import Signal, SignalType

DAY = date(2026, 10, 1)
T = lambda h, m=0: datetime(2026, 10, 1, h, m, tzinfo=timezone.utc).isoformat()


def _strat(name, symbol):
    s = Mock()
    s.name, s.symbol, s.is_active = name, symbol, True
    return s


def _closes(symbol):
    base = {"BTCUSDT": 60_000.0, "ETHUSDT": 2_000.0}[symbol]
    idx = pd.date_range("2026-09-28", periods=4, freq="1D", tz="UTC")   # 09-28 .. 10-01
    growth = {"BTCUSDT": [1.0, 1.0, 1.0, 1.02], "ETHUSDT": [1.0, 1.0, 1.0, 0.95]}[symbol]
    return pd.Series([base * g for g in growth], index=idx)


@pytest.fixture
def reporter(temp_db, tmp_path, monkeypatch):
    from daily_report import DailyReporter
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 1_000.0)
    monkeypatch.setattr(config, "SYMBOLS", ["BTCUSDT", "ETHUSDT"])
    strats = [_strat("A", "BTCUSDT"), _strat("A@ETH", "ETHUSDT")]
    books = {
        "aprende": ("main", PortfolioManager(Mock(), strats, book="main", simulate_fills=True,
                                             capital_base=100)),
        "lab": ("lab", PortfolioManager(Mock(), strats, book="lab", simulate_fills=True,
                                        capital_base=2_000)),
        "baseline": ("baseline", PortfolioManager(Mock(), strats, book="baseline", simulate_fills=True,
                                                  capital_base=2_000)),
    }
    now = {"t": datetime(2026, 10, 2, 0, 10, tzinfo=timezone.utc)}
    rep = DailyReporter(out_dir=str(tmp_path), books=books,
                        prices_fn=lambda: {"BTCUSDT": 61_200.0, "ETHUSDT": 1_900.0},
                        closes_fn=_closes, clock=lambda: now["t"])
    rep._now = now
    return rep


def _history_of_the_day():
    for book, sym, name, pnl in (("main", "BTCUSDT", "A", 3.0), ("main", "ETHUSDT", "A@ETH", -1.0),
                                 ("lab", "ETHUSDT", "A@ETH", -4.0)):
        db.record_trade(strategy_name=name, symbol=sym, side="LONG", entry_price=1.0, exit_price=1.0,
                        quantity=1.0, pnl=pnl, pnl_pct=0.01, fees_paid=0.1, entry_time=T(2),
                        exit_time=T(5), duration_hours=3.0, exit_reason="TAKE_PROFIT",
                        entry_features={}, book=book, closed_at=T(5))
    m = {"trades": 40}
    day_before = (datetime(2026, 9, 30, 12, tzinfo=timezone.utc)).isoformat()
    db.record_strategy_evaluation(day_before, "Turtle_Breakout", "EN_PRUEBA", 0.3, "pocos trades", m)
    db.record_strategy_evaluation(T(14), "Turtle_Breakout", "VIABLE", 0.8, "gana en 3/4 ventanas", m)
    db.upsert_strategy_status("Turtle_Breakout", "VIABLE", 0.8, [], ["LONG"], "gana en 3/4 ventanas", m, ts=T(14))
    db.record_strategy_evaluation(day_before, "Breakout@ETH", "CONDICIONAL", 0.5, "solo lateral", m)
    # added and discarded again within the day: only the net change is reported
    db.record_strategy_evaluation(T(1), "Supertrend@ETH", "VIABLE", 0.7, "gana", m)
    db.record_strategy_evaluation(T(13), "Supertrend@ETH", "DESCARTADA", 0.0, "pierde", m)
    db.record_strategy_evaluation(T(14), "Breakout@ETH", "DESCARTADA", 0.0, "pierde de forma consistente", m)
    db.upsert_strategy_status("Breakout@ETH", "DESCARTADA", 0.0, [], [], "pierde de forma consistente", m, ts=T(14))
    db.record_learning_audit(ts=T(15), strategy_name="EMA5_Momentum", decision="applied",
                             reason="validation PF 1.09->1.30", param="ema_period", old_value=5,
                             new_value=4, book="main")
    for i in range(3):
        db.record_signal(book="main", strategy_name="A", candle_ts=f"{T(i + 6)}", signal_type="BUY",
                         confidence=0.7, ml_confidence=0.6, price=1.0, acted=False,
                         reason="riesgo: maximo de 2 posiciones abiertas", recorded_at=T(i + 6))


def test_report_covers_results_coins_strategies_learning_and_risk(reporter):
    _history_of_the_day()
    text = reporter.build(DAY)
    assert "INFORME DIARIO" in text and "2026-10-01" in text
    assert "BTC buy & hold" in text and "+2.00%" in text          # 60,000 -> 61,200
    assert "Canasta" in text and "-1.50%" in text                 # (+2% - 5%) / 2
    assert "AGREGADAS" in text and "Turtle_Breakout" in text and "EN_PRUEBA -> VIABLE" in text
    assert "QUITADAS" in text and "CONDICIONAL -> DESCARTADA" in text and "valor 0" in text
    added = text.split("AGREGADAS")[1].split("QUITADAS")[0]
    assert "Supertrend@ETH" not in added and "nueva -> DESCARTADA" in text
    assert "ema_period 5 -> 4" in text
    assert "3 x riesgo: maximo de 2 posiciones abiertas" in text
    btc_row = next(l for l in text.splitlines() if l.startswith("BTC "))
    eth_row = next(l for l in text.splitlines() if l.startswith("ETH "))
    assert "+3.00" in btc_row and "-1.00" in eth_row and "-4.00" in eth_row


def test_report_is_written_once_per_utc_day(reporter, tmp_path):
    _history_of_the_day()
    path = reporter.write_if_due()
    assert path and os.path.basename(path) == "informe_diario_2026-10-01.txt"
    assert reporter.write_if_due() is None                        # same day: nothing new
    reporter._now["t"] += timedelta(days=1)
    assert os.path.basename(reporter.write_if_due()) == "informe_diario_2026-10-02.txt"
    assert db.get_meta("daily_report:latest").endswith("informe_diario_2026-10-02.txt")


def test_day_change_uses_the_previous_report_equity(reporter):
    first = reporter.build(DAY, remember=True)
    assert "sin informe anterior" in first
    reporter._now["t"] += timedelta(days=1)
    second = reporter.build(DAY + timedelta(days=1))
    assert "vs informe anterior" in second


def test_report_never_breaks_on_missing_prices(reporter):
    reporter.closes_fn = lambda sym: pd.Series(dtype=float)
    text = reporter.build(DAY)
    assert "sin datos" in text


def test_the_bot_writes_the_report_from_its_learning_loop(temp_db, tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    bot = main.TradingBot.__new__(main.TradingBot)
    strats = [_strat("A", "BTCUSDT")]
    bot.book, bot._prices, bot._current_price = "main", {"BTCUSDT": 61_200.0}, 61_200.0
    bot.portfolio = PortfolioManager(Mock(), strats, book="main", simulate_fills=True, capital_base=100)
    bot.lab_portfolio = PortfolioManager(Mock(), strats, book="lab", simulate_fills=True)
    bot.baseline_portfolio = PortfolioManager(Mock(), strats, book="baseline", simulate_fills=True)
    bot.client = Mock()
    bot.client.get_latest_candles.side_effect = lambda sym, iv, limit=10: pd.DataFrame(
        {"close": _closes(sym).to_numpy()}, index=_closes(sym).index)
    bot.reporter = bot._build_reporter()
    assert set(bot.reporter.books) == {"aprende", "lab", "baseline"}
    path = bot._write_daily_report()
    assert path and os.path.dirname(path) == os.path.join(str(tmp_path), "reports")
    assert bot._write_daily_report() is None
    bot.client.get_latest_candles.side_effect = RuntimeError("binance down")
    db.set_meta("daily_report:last_day", "2000-01-01")
    assert bot._write_daily_report()               # never breaks the learning loop


def test_dashboard_shows_the_latest_daily_report(temp_db, tmp_path):
    import dash
    from dashboard import app as dash_app
    from tests.test_dashboard import _dump
    assert "todav" in _dump(dash_app._render_daily_report()).lower()
    for day, text in (("2026-10-01", "INFORME UNO"), ("2026-10-02", "INFORME DOS")):
        p = tmp_path / f"informe_diario_{day}.txt"
        p.write_text(text, encoding="utf-8")
    db.set_meta("daily_report:latest", str(tmp_path / "informe_diario_2026-10-02.txt"))
    out = _dump(dash_app._render_daily_report())
    assert "INFORME DOS" in out and "2026-10-01" in out          # latest shown, older selectable
    assert "INFORME UNO" in _dump(dash_app.show_daily_report(str(tmp_path / "informe_diario_2026-10-01.txt")))
    assert dash_app.render_tab_for("tab-report", "interval-refresh") is dash.no_update
