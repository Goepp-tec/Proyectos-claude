"""DATA_DIR keeps the SQLite DB (+ its -wal/-shm files) and the log in one mountable folder."""

import importlib
import os

import config


def test_data_dir_moves_db_and_log(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    importlib.reload(config)
    try:
        assert config.DB_PATH == os.path.join(str(tmp_path), "trading_bot.db")
        assert config.LOG_FILE == os.path.join(str(tmp_path), "trading_bot.log")
    finally:
        monkeypatch.delenv("DATA_DIR")
        importlib.reload(config)


def test_empty_data_dir_falls_back_to_project_folder(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "")
    importlib.reload(config)
    try:
        assert os.path.isabs(config.DB_PATH)
        assert os.path.dirname(config.DB_PATH) == os.path.dirname(os.path.abspath(config.__file__))
    finally:
        monkeypatch.delenv("DATA_DIR")
        importlib.reload(config)


def test_default_paths_unchanged_without_data_dir():
    app_dir = os.path.dirname(os.path.abspath(config.__file__))
    assert os.path.dirname(os.path.abspath(config.DB_PATH)) == app_dir
    assert os.path.dirname(os.path.abspath(config.LOG_FILE)) == app_dir
