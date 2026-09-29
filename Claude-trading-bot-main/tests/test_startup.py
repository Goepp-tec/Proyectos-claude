"""A0: first start with an empty database must not crash."""

from unittest.mock import patch

import database as db


def test_main_on_fresh_database_creates_schema_before_live_since(fresh_db_path):
    import main

    with patch.object(main, "TradingBot") as fake_bot:
        main.main()   # used to raise sqlite3.OperationalError: no such table: bot_metadata

    assert db.get_live_since() is not None
    fake_bot.return_value.run.assert_called_once()
