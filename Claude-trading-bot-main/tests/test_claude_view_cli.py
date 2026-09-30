"""scripts/claude_view.py: what a scheduled Claude review reads, and how it leaves its view."""

import json
from datetime import datetime, timedelta, timezone

import pytest

import config
import database as db

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def test_context_lists_recent_decisions_positions_news_and_calendar(temp_db, monkeypatch):
    from news_data import _ensure_tables
    from scripts.claude_view import build_context
    monkeypatch.setattr(config, "SYMBOLS", ["BTCUSDT", "ETHUSDT"])
    db.set_meta("trading_mode", "OBSERVE")
    db.record_signal(book="observe", strategy_name="Turtle_Breakout@ETH", candle_ts="c1",
                     signal_type="BUY", confidence=0.7, ml_confidence=0.6, price=2000.0, acted=False,
                     reason="confirmaciones: faltan confirmaciones (1 a favor, 2 en contra)",
                     recorded_at=(NOW - timedelta(hours=2)).isoformat())
    db.open_position(strategy_name="EMA5_Momentum", symbol="BTCUSDT", side="LONG", entry_price=60000.0,
                     quantity=0.001, stop_loss=58000.0, take_profit=64000.0, order_id="x",
                     ml_confidence=0.6, metadata={"confirmaciones": {"neto": 3}}, book="observe",
                     entry_time=(NOW - timedelta(hours=5)).isoformat())
    _ensure_tables()
    db.get_conn().execute("INSERT INTO news_items (ts, source, title, link, coins, tone, severe) "
                          "VALUES (?, 'coindesk', 'Ethereum ETF sees record inflows', 'l', 'ETH', 0.8, 0)",
                          ((NOW - timedelta(hours=1)).isoformat(),))
    db.get_conn().execute("INSERT INTO macro_events VALUES (?, 'USD', 'FOMC Statement', 'High')",
                          ((NOW + timedelta(hours=6)).isoformat(),))
    db.get_conn().commit()
    text = build_context(NOW)
    for expected in ("Turtle_Breakout@ETH", "faltan confirmaciones", "EMA5_Momentum", "neto 3",
                     "Ethereum ETF sees record inflows", "FOMC Statement", "BTC", "ETH"):
        assert expected in text, expected


def test_set_and_show_views_from_the_command_line(temp_db, capsys):
    from scripts.claude_view import main
    main(["set", "--coin", "sol", "--long", "bloquear", "--short", "neutral", "--hours", "6",
          "--nota", "exploit en un DEX grande"])
    main(["show"])
    out = capsys.readouterr().out
    assert "SOL" in out and "bloquear" in out and "exploit" in out and "vigente" in out
    with pytest.raises(SystemExit):
        main(["set", "--coin", "BTC", "--long", "comprar_todo"])      # only the 3 allowed values
