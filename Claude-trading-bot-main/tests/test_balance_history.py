"""Equity curve: balance rows recorded on the first day of a run are shown."""

import database as db


def test_first_day_balance_rows_are_not_hidden_by_the_date_format(temp_db):
    """recorded_at is 'YYYY-MM-DD HH:MM:SS' (SQLite) and live_since is ISO with a
    'T': compared as text, ' ' < 'T' hid every row of the first day."""
    conn = db.get_conn()
    conn.execute("INSERT OR REPLACE INTO bot_metadata (key, value) VALUES "
                 "('live_since', '2026-09-30T03:15:00.123456+00:00')")
    for ts, bal in (("2026-09-30 03:10:00", 99.0),          # before the run started
                    ("2026-09-30 03:57:00", 100.0), ("2026-09-30 04:10:00", 100.5)):
        conn.execute("INSERT INTO balance_history (total_balance, realized_pnl, unrealized_pnl, "
                     "strategy_breakdown, book, recorded_at) VALUES (?, 0, 0, '{}', 'observe', ?)", (bal, ts))
    conn.commit()
    rows = db.get_balance_history(book="observe")
    assert [r["total_balance"] for r in rows] == [100.0, 100.5]
