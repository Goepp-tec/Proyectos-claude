"""Shared pytest fixtures."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database as db


def _reset_conn():
    conn = getattr(db._local, "conn", None)
    if conn is not None:
        conn.close()
    db._local.conn = None


@pytest.fixture
def fresh_db_path(tmp_path, monkeypatch):
    """Point the bot at an empty, never-initialised SQLite file."""
    path = str(tmp_path / "test_trading_bot.db")
    monkeypatch.setattr(config, "DB_PATH", path)
    _reset_conn()
    yield path
    _reset_conn()


@pytest.fixture
def temp_db(fresh_db_path):
    """Empty SQLite database with the full schema created."""
    db.init_db()
    yield fresh_db_path
