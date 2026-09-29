"""Deployed databases (without the 'book' column) migrate without losing rows."""

import database as db


def test_migration_keeps_existing_rows_in_main_book(fresh_db_path):
    import sqlite3
    from tests.conftest import _reset_conn
    _reset_conn()
    old = sqlite3.connect(fresh_db_path)      # schema as deployed before this change
    old.execute("""CREATE TABLE positions (id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_name TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
        entry_price REAL NOT NULL, quantity REAL NOT NULL, stop_loss REAL,
        take_profit REAL, entry_time TEXT NOT NULL, order_id TEXT,
        status TEXT DEFAULT 'OPEN', ml_confidence REAL DEFAULT 0.5,
        metadata TEXT DEFAULT '{}')""")
    old.execute("""INSERT INTO positions (strategy_name, symbol, side, entry_price,
        quantity, stop_loss, take_profit, entry_time) VALUES
        ('EMA5_Momentum','BTCUSDT','SHORT',82881.49,0.00345,86899.06,75568.9,'2026-09-29T03:05:59')""")
    old.commit()
    old.close()

    db.init_db()
    rows = db.get_open_positions("EMA5_Momentum")
    assert len(rows) == 1 and rows[0]["book"] == "main"
