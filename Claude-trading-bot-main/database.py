"""SQLite database layer for the trading bot.

Changes:
  - Added utc_now and utc_now_iso utility for timezone-aware datetime handling
  - Replaced deprecated datetime.utcnow() calls
  - Added is_backtest column to trades table to distinguish backtest from live trades
  - Added live_since tracking to know when real trading started
  - Added clear_old_data function to reset database
"""

import sqlite3
import json
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any

import config
from utils import utc_now, utc_now_iso

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS strategies (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            is_active   INTEGER DEFAULT 0,
            capital     REAL DEFAULT 0,
            params      TEXT DEFAULT '{}',
            backtest_cagr REAL DEFAULT 0,
            backtest_win_rate REAL DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS positions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name   TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            side            TEXT NOT NULL,       -- LONG or SHORT
            entry_price     REAL NOT NULL,
            quantity        REAL NOT NULL,
            stop_loss       REAL,
            take_profit     REAL,
            entry_time      TEXT NOT NULL,
            order_id        TEXT,
            status          TEXT DEFAULT 'OPEN', -- OPEN, CLOSED
            ml_confidence   REAL DEFAULT 0.5,
            metadata        TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name   TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            side            TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            exit_price      REAL NOT NULL,
            quantity        REAL NOT NULL,
            pnl             REAL NOT NULL,
            pnl_pct         REAL NOT NULL,
            fees_paid       REAL NOT NULL,
            entry_time      TEXT NOT NULL,
            exit_time       TEXT NOT NULL,
            duration_hours  REAL NOT NULL,
            exit_reason     TEXT,               -- TAKE_PROFIT, STOP_LOSS, SIGNAL, MANUAL
            entry_features  TEXT DEFAULT '{}',  -- JSON of indicator values at entry
            closed_at       TEXT DEFAULT (datetime('now')),
            is_backtest    INTEGER DEFAULT 0   -- 0=live trade, 1=backtest (not shown in history)
        );

        CREATE TABLE IF NOT EXISTS journal_entries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        INTEGER NOT NULL REFERENCES trades(id),
            strategy_name   TEXT NOT NULL,
            entry_price     REAL,
            exit_price      REAL,
            pnl             REAL,
            pnl_pct         REAL,
            side            TEXT,
            duration_hours  REAL,
            market_regime   TEXT,
            setup_summary   TEXT,
            outcome_analysis TEXT,
            reflection      TEXT,
            lessons         TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS balance_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            total_balance   REAL NOT NULL,
            realized_pnl    REAL NOT NULL,
            unrealized_pnl  REAL NOT NULL,
            strategy_breakdown TEXT DEFAULT '{}',
            recorded_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS ml_features (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        INTEGER REFERENCES trades(id),
            strategy_name   TEXT NOT NULL,
            features        TEXT NOT NULL,       -- JSON array
            outcome         REAL NOT NULL,       -- 1 = win, 0 = loss
            pnl_pct         REAL NOT NULL,
            recorded_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS strategy_performance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name   TEXT NOT NULL,
            date            TEXT NOT NULL,
            capital         REAL NOT NULL,
            daily_pnl       REAL DEFAULT 0,
            cumulative_pnl  REAL DEFAULT 0,
            win_rate        REAL DEFAULT 0,
            total_trades    INTEGER DEFAULT 0,
            open_positions  INTEGER DEFAULT 0,
            UNIQUE(strategy_name, date)
        );
    """)
    conn.commit()
    
    # ─── Database migrations ──────────────────────────────────────────────────
    # Add is_backtest column if it doesn't exist (for existing databases)
    try:
        conn.execute("SELECT is_backtest FROM trades LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE trades ADD COLUMN is_backtest INTEGER DEFAULT 0")
        conn.commit()

    # One row per strategy signal per closed candle (acted on or not, and why).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            book            TEXT NOT NULL,
            strategy_name   TEXT NOT NULL,
            candle_ts       TEXT NOT NULL,
            signal_type     TEXT NOT NULL,
            confidence      REAL,
            ml_confidence   REAL,
            price           REAL,
            acted           INTEGER NOT NULL,
            reason          TEXT,
            recorded_at     TEXT DEFAULT (datetime('now'))
        )
    """)

    # Learning audit: every parameter proposal and what happened to it.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learning_audit (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            strategy_name   TEXT NOT NULL,
            param           TEXT,
            old_value       REAL,
            new_value       REAL,
            metrics_before  TEXT DEFAULT '{}',   -- JSON
            metrics_after   TEXT DEFAULT '{}',   -- JSON
            decision        TEXT NOT NULL,       -- applied / rejected / rollback
            reason          TEXT,
            book            TEXT,                -- book of the learning strategy
            evaluate_after  TEXT,                -- applied: when to check for rollback
            evaluated       INTEGER DEFAULT 0,   -- applied: 1 once kept or rolled back
            learner_equity  REAL,                -- equity snapshots when applied
            baseline_equity REAL
        )
    """)

    # Free market data (market_data.py): futures positioning, funding, Fear & Greed.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            symbol  TEXT NOT NULL,
            metric  TEXT NOT NULL,
            ts      TEXT NOT NULL,      -- ISO UTC, when the value was published
            value   REAL NOT NULL,
            PRIMARY KEY (symbol, metric, ts)
        )
    """)

    # Strategy evaluator: current rating per strategy + history of evaluations.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_status (
            strategy_name   TEXT PRIMARY KEY,
            status          TEXT NOT NULL,       -- VIABLE / CONDICIONAL / EN_PRUEBA / DESCARTADA
            score           REAL NOT NULL,       -- 0..1 (0 = discarded)
            allowed_regimes TEXT DEFAULT '[]',   -- JSON
            allowed_sides   TEXT DEFAULT '[]',   -- JSON
            reason          TEXT,
            metrics         TEXT DEFAULT '{}',   -- JSON
            updated_at      TEXT,
            discarded_at    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_evaluations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            strategy_name   TEXT NOT NULL,
            status          TEXT NOT NULL,
            score           REAL NOT NULL,
            reason          TEXT,
            metrics         TEXT DEFAULT '{}'
        )
    """)

    # Paper "books": 'main' (what the bot trades), 'observe' (observation-mode
    # theoretical trades), 'baseline' (frozen-parameter shadow copies).
    for table in ("positions", "trades", "balance_history"):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if "book" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN book TEXT DEFAULT 'main'")
    conn.commit()

    # Add live_since table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def clear_old_data():
    """
    Clear all trading data to start fresh.
    Use this when you want to reset the bot and start new.
    """
    conn = get_conn()
    
    # Delete all trades (keep strategies)
    conn.execute("DELETE FROM trades")
    
    # Delete all positions
    conn.execute("DELETE FROM positions")
    
    # Delete all journal entries
    conn.execute("DELETE FROM journal_entries")
    
    # Delete all balance history
    conn.execute("DELETE FROM balance_history")
    
    # Delete all ML features
    conn.execute("DELETE FROM ml_features")
    
    # Delete all performance history
    conn.execute("DELETE FROM strategy_performance")
    
    # Update live_since to now
    conn.execute("""
        INSERT OR REPLACE INTO bot_metadata (key, value, updated_at)
        VALUES ('live_since', ?, datetime('now'))
    """, (utc_now_iso(),))
    
    conn.commit()


def get_live_since() -> Optional[str]:
    """Get the date when live trading started."""
    row = get_conn().execute(
        "SELECT value FROM bot_metadata WHERE key='live_since'"
    ).fetchone()
    return row[0] if row else None


def set_live_since():
    """Set the live trading start date if not already set."""
    existing = get_live_since()
    if not existing:
        conn = get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO bot_metadata (key, value, updated_at)
            VALUES ('live_since', ?, datetime('now'))
        """, (utc_now_iso(),))
        conn.commit()

def get_meta(key: str) -> Optional[str]:
    row = get_conn().execute("SELECT value FROM bot_metadata WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(key: str, value: str):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO bot_metadata (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
    """, (key, value))
    conn.commit()


def record_signal(book: str, strategy_name: str, candle_ts: str, signal_type: str,
                  confidence: float, ml_confidence: float, price: float,
                  acted: bool, reason: str, recorded_at: str = None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO signal_log (book, strategy_name, candle_ts, signal_type, confidence,
                                ml_confidence, price, acted, reason, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, datetime('now')))
    """, (book, strategy_name, candle_ts, signal_type, confidence, ml_confidence,
          price, int(acted), reason, recorded_at))
    conn.commit()


def get_signals(book: Optional[str] = None, limit: int = 200) -> List[Dict]:
    extra, params = _book_filter(book, ())
    rows = get_conn().execute(
        f"SELECT * FROM signal_log WHERE 1=1{extra} ORDER BY id DESC LIMIT ?",
        params + (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def record_learning_audit(ts: str, strategy_name: str, decision: str, reason: str,
                          param: str = None, old_value: float = None, new_value: float = None,
                          metrics_before: dict = None, metrics_after: dict = None,
                          book: str = None, evaluate_after: str = None,
                          learner_equity: float = None, baseline_equity: float = None) -> int:
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO learning_audit (ts, strategy_name, param, old_value, new_value,
            metrics_before, metrics_after, decision, reason, book, evaluate_after,
            learner_equity, baseline_equity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, strategy_name, param, old_value, new_value,
          json.dumps(metrics_before or {}), json.dumps(metrics_after or {}),
          decision, reason, book, evaluate_after, learner_equity, baseline_equity))
    conn.commit()
    return cur.lastrowid


def get_learning_audit(strategy_name: str = None, decision: str = None,
                       limit: int = 500) -> List[Dict]:
    where, params = "WHERE 1=1", ()
    if strategy_name:
        where, params = where + " AND strategy_name=?", params + (strategy_name,)
    if decision:
        where, params = where + " AND decision=?", params + (decision,)
    rows = get_conn().execute(
        f"SELECT * FROM learning_audit {where} ORDER BY id DESC LIMIT ?", params + (limit,)
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["metrics_before"] = json.loads(d.get("metrics_before") or "{}")
        d["metrics_after"] = json.loads(d.get("metrics_after") or "{}")
        result.append(d)
    return result


def get_pending_learning_changes() -> List[Dict]:
    rows = get_conn().execute(
        "SELECT * FROM learning_audit WHERE decision='applied' AND evaluated=0 ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def mark_learning_change_evaluated(audit_id: int):
    conn = get_conn()
    conn.execute("UPDATE learning_audit SET evaluated=1 WHERE id=?", (audit_id,))
    conn.commit()


def _status_row(row) -> Optional[Dict]:
    if not row:
        return None
    d = dict(row)
    for k in ("allowed_regimes", "allowed_sides", "metrics"):
        d[k] = json.loads(d.get(k) or ("{}" if k == "metrics" else "[]"))
    return d


def upsert_strategy_status(strategy_name: str, status: str, score: float,
                           allowed_regimes: list, allowed_sides: list, reason: str,
                           metrics: dict, ts: str = None):
    ts = ts or utc_now_iso()
    conn = get_conn()
    conn.execute("""
        INSERT INTO strategy_status (strategy_name, status, score, allowed_regimes,
            allowed_sides, reason, metrics, updated_at, discarded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy_name) DO UPDATE SET
            status=excluded.status, score=excluded.score,
            allowed_regimes=excluded.allowed_regimes, allowed_sides=excluded.allowed_sides,
            reason=excluded.reason, metrics=excluded.metrics, updated_at=excluded.updated_at,
            discarded_at=COALESCE(strategy_status.discarded_at, excluded.discarded_at)
    """, (strategy_name, status, score, json.dumps(allowed_regimes), json.dumps(allowed_sides),
          reason, json.dumps(metrics), ts, ts if status == "DESCARTADA" else None))
    conn.commit()


def get_strategy_status(strategy_name: str) -> Optional[Dict]:
    return _status_row(get_conn().execute(
        "SELECT * FROM strategy_status WHERE strategy_name=?", (strategy_name,)).fetchone())


def get_all_strategy_status() -> List[Dict]:
    rows = get_conn().execute("SELECT * FROM strategy_status ORDER BY score DESC").fetchall()
    return [_status_row(r) for r in rows]


def record_strategy_evaluation(ts: str, strategy_name: str, status: str, score: float,
                               reason: str, metrics: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO strategy_evaluations (ts, strategy_name, status, score, reason, metrics)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ts, strategy_name, status, score, reason, json.dumps(metrics)))
    conn.commit()


def get_strategy_evaluations(strategy_name: str = None, limit: int = 500) -> List[Dict]:
    where, params = ("WHERE strategy_name=?", (strategy_name,)) if strategy_name else ("", ())
    rows = get_conn().execute(
        f"SELECT * FROM strategy_evaluations {where} ORDER BY id DESC LIMIT ?", params + (limit,)
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d.get("metrics") or "{}")
        result.append(d)
    return result


def count_trades_since(strategy_name: str, since_iso: str, book: str) -> int:
    row = get_conn().execute(
        "SELECT COUNT(*) FROM trades WHERE strategy_name=? AND book=? AND is_backtest=0 "
        "AND closed_at > ?", (strategy_name, book, since_iso)
    ).fetchone()
    return int(row[0] or 0)


def upsert_strategy(name: str, capital: float, params: dict,
                    backtest_cagr: float = 0.0, backtest_win_rate: float = 0.0,
                    is_active: bool = False):
    conn = get_conn()
    conn.execute("""
        INSERT INTO strategies (name, capital, params, backtest_cagr, backtest_win_rate, is_active)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            capital = excluded.capital,
            params  = excluded.params,
            backtest_cagr = excluded.backtest_cagr,
            backtest_win_rate = excluded.backtest_win_rate,
            is_active = excluded.is_active
    """, (name, capital, json.dumps(params), backtest_cagr, backtest_win_rate, int(is_active)))
    conn.commit()


def get_strategy(name: str) -> Optional[Dict]:
    row = get_conn().execute("SELECT * FROM strategies WHERE name=?", (name,)).fetchone()
    if row:
        d = dict(row)
        d["params"] = json.loads(d["params"])
        return d
    return None


def get_active_strategies() -> List[Dict]:
    rows = get_conn().execute("SELECT * FROM strategies WHERE is_active=1").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["params"] = json.loads(d["params"])
        result.append(d)
    return result


def update_strategy_capital(name: str, capital: float):
    conn = get_conn()
    conn.execute("UPDATE strategies SET capital=? WHERE name=?", (capital, name))
    conn.commit()


def update_strategy_params(name: str, params: dict):
    conn = get_conn()
    conn.execute("UPDATE strategies SET params=? WHERE name=?", (json.dumps(params), name))
    conn.commit()


# ─── Position helpers ─────────────────────────────────────────────────────────

def _book_filter(book: Optional[str], params: tuple) -> tuple:
    """SQL fragment + params restricting to one book (book=None -> all books)."""
    if book is None:
        return "", params
    return " AND book=?", params + (book,)


def open_position(strategy_name: str, symbol: str, side: str,
                  entry_price: float, quantity: float,
                  stop_loss: float, take_profit: float,
                  order_id: str = "", ml_confidence: float = 0.5,
                  metadata: dict = None, book: str = "main",
                  entry_time: str = None) -> int:
    conn = get_conn()
    cursor = conn.execute("""
        INSERT INTO positions
        (strategy_name, symbol, side, entry_price, quantity, stop_loss,
         take_profit, entry_time, order_id, ml_confidence, metadata, book)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (strategy_name, symbol, side, entry_price, quantity, stop_loss,
          take_profit, entry_time or utc_now_iso(), order_id,
          ml_confidence, json.dumps(metadata or {}), book))
    conn.commit()
    return cursor.lastrowid


def get_open_positions(strategy_name: str = None, book: Optional[str] = "main") -> List[Dict]:
    conn = get_conn()
    if strategy_name:
        extra, params = _book_filter(book, (strategy_name,))
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='OPEN' AND strategy_name=?" + extra,
            params
        ).fetchall()
    else:
        extra, params = _book_filter(book, ())
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='OPEN'" + extra, params
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        result.append(d)
    return result


def close_position(position_id: int):
    conn = get_conn()
    conn.execute("UPDATE positions SET status='CLOSED' WHERE id=?", (position_id,))
    conn.commit()


# ─── Trade helpers ────────────────────────────────────────────────────────────

def record_trade(strategy_name: str, symbol: str, side: str,
                 entry_price: float, exit_price: float, quantity: float,
                 pnl: float, pnl_pct: float, fees_paid: float,
                 entry_time: str, exit_time: str, duration_hours: float,
                 exit_reason: str, entry_features: dict = None,
                 is_backtest: bool = False, book: str = "main",
                 closed_at: str = None) -> int:
    """Record a trade. Set is_backtest=True for backtest trades (not shown in history)."""
    conn = get_conn()
    cursor = conn.execute("""
        INSERT INTO trades
        (strategy_name, symbol, side, entry_price, exit_price, quantity,
         pnl, pnl_pct, fees_paid, entry_time, exit_time, duration_hours,
         exit_reason, entry_features, is_backtest, book, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE(?, datetime('now')))
    """, (strategy_name, symbol, side, entry_price, exit_price, quantity,
          pnl, pnl_pct, fees_paid, entry_time, exit_time, duration_hours,
          exit_reason, json.dumps(entry_features or {}), 1 if is_backtest else 0,
          book, closed_at))
    conn.commit()
    return cursor.lastrowid


def get_trades(strategy_name: str = None, limit: int = 500, include_backtest: bool = False,
               book: Optional[str] = "main") -> List[Dict]:
    """
    Get trades from the database.

    Args:
        strategy_name: Filter by strategy (optional)
        limit: Maximum number of trades to return
        include_backtest: If False (default), exclude backtest trades
        book: paper book to read ('main' by default, None = all books)
    """
    conn = get_conn()
    where, params = "WHERE 1=1", ()
    if strategy_name:
        where, params = where + " AND strategy_name=?", (strategy_name,)
    if not include_backtest:
        where += " AND is_backtest=0"
    extra, params = _book_filter(book, params)
    rows = conn.execute(
        f"SELECT * FROM trades {where}{extra} ORDER BY closed_at DESC, id DESC LIMIT ?",
        params + (limit,)
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["entry_features"] = json.loads(d.get("entry_features") or "{}")
        result.append(d)
    return result


def get_trade_stats(strategy_name: str = None, include_backtest: bool = False,
                    book: Optional[str] = "main") -> Dict:
    """Get trade statistics. Excludes backtest trades by default."""
    conn = get_conn()
    where, params = "WHERE 1=1", ()
    if strategy_name:
        where, params = where + " AND strategy_name=?", (strategy_name,)
    if not include_backtest:
        where += " AND is_backtest=0"
    extra, params = _book_filter(book, params)
    where += extra

    query = f"""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(pnl) as total_pnl,
            AVG(pnl_pct) as avg_pnl_pct,
            MAX(pnl) as best_trade,
            MIN(pnl) as worst_trade,
            AVG(duration_hours) as avg_duration_hours,
            SUM(fees_paid) as total_fees
        FROM trades {where}
    """
    row = conn.execute(query, params).fetchone()
    if row:
        d = dict(row)
        d["win_rate"] = (d["wins"] / d["total_trades"]) if d["total_trades"] > 0 else 0
        return d
    return {}


# ─── Journal helpers ──────────────────────────────────────────────────────────

def record_journal_entry(trade_id: int, strategy_name: str,
                         entry_price: float, exit_price: float,
                         pnl: float, pnl_pct: float, side: str,
                         duration_hours: float, market_regime: str,
                         setup_summary: str, outcome_analysis: str,
                         reflection: str, lessons: str):
    conn = get_conn()
    conn.execute("""
        INSERT INTO journal_entries
        (trade_id, strategy_name, entry_price, exit_price, pnl, pnl_pct,
         side, duration_hours, market_regime, setup_summary, outcome_analysis,
         reflection, lessons)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (trade_id, strategy_name, entry_price, exit_price, pnl, pnl_pct,
          side, duration_hours, market_regime, setup_summary, outcome_analysis,
          reflection, lessons))
    conn.commit()


def get_journal_entries(strategy_name: str = None, limit: int = 100, include_backtest: bool = False) -> List[Dict]:
    """Get journal entries. Excludes entries for backtest trades by default."""
    conn = get_conn()
    if strategy_name:
        if include_backtest:
            rows = conn.execute(
                "SELECT * FROM journal_entries WHERE strategy_name=? ORDER BY created_at DESC LIMIT ?",
                (strategy_name, limit)
            ).fetchall()
        else:
            # Join with trades to filter out backtest trades
            rows = conn.execute("""
                SELECT j.* FROM journal_entries j
                JOIN trades t ON j.trade_id = t.id
                WHERE j.strategy_name=? AND t.is_backtest=0
                ORDER BY j.created_at DESC LIMIT ?
            """, (strategy_name, limit)).fetchall()
    else:
        if include_backtest:
            rows = conn.execute(
                "SELECT * FROM journal_entries ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute("""
                SELECT j.* FROM journal_entries j
                JOIN trades t ON j.trade_id = t.id
                WHERE t.is_backtest=0
                ORDER BY j.created_at DESC LIMIT ?
            """, (limit,)).fetchall()
    return [dict(row) for row in rows]


# ─── Balance helpers ──────────────────────────────────────────────────────────

def record_balance(total_balance: float, realized_pnl: float,
                   unrealized_pnl: float, strategy_breakdown: dict = None,
                   book: str = "main"):
    conn = get_conn()
    conn.execute("""
        INSERT INTO balance_history (total_balance, realized_pnl, unrealized_pnl, strategy_breakdown, book)
        VALUES (?, ?, ?, ?, ?)
    """, (total_balance, realized_pnl, unrealized_pnl,
          json.dumps(strategy_breakdown or {}), book))
    conn.commit()


def get_balance_history(days: int = 30, include_backtest: bool = False,
                        book: str = "main") -> List[Dict]:
    """
    Get balance history for equity curve.
    
    Args:
        days: Number of days to look back
        include_backtest: If False (default), only return data from when live trading started
    """
    conn = get_conn()
    
    # Get live_since date if we should filter
    live_since = None if include_backtest else get_live_since()
    
    if live_since:
        # Filter to only include data from live trading start. Compare as dates:
        # recorded_at is 'YYYY-MM-DD HH:MM:SS' and live_since ISO with a 'T', so a
        # text comparison hid every row of the first day.
        rows = conn.execute("""
            SELECT * FROM balance_history
            WHERE datetime(recorded_at) >= datetime(?) AND book=?
            ORDER BY recorded_at ASC
        """, (live_since, book)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM balance_history
            WHERE recorded_at >= datetime('now', ? || ' days') AND book=?
            ORDER BY recorded_at ASC
        """, (f"-{days}", book)).fetchall()
    
    result = []
    for row in rows:
        d = dict(row)
        d["strategy_breakdown"] = json.loads(d.get("strategy_breakdown") or "{}")
        result.append(d)
    return result


def get_latest_balance(book: str = "main") -> Optional[Dict]:
    row = get_conn().execute(
        "SELECT * FROM balance_history WHERE book=? ORDER BY recorded_at DESC, id DESC LIMIT 1",
        (book,)
    ).fetchone()
    if row:
        d = dict(row)
        d["strategy_breakdown"] = json.loads(d.get("strategy_breakdown") or "{}")
        return d
    return None


# ─── ML Feature helpers ───────────────────────────────────────────────────────

def record_ml_features(trade_id: int, strategy_name: str,
                       features: List[float], outcome: float, pnl_pct: float):
    conn = get_conn()
    conn.execute("""
        INSERT INTO ml_features (trade_id, strategy_name, features, outcome, pnl_pct)
        VALUES (?, ?, ?, ?, ?)
    """, (trade_id, strategy_name, json.dumps(features), outcome, pnl_pct))
    conn.commit()


def get_ml_features(strategy_name: str, limit: int = 500) -> List[Dict]:
    rows = get_conn().execute("""
        SELECT * FROM ml_features WHERE strategy_name=?
        ORDER BY recorded_at DESC LIMIT ?
    """, (strategy_name, limit)).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["features"] = json.loads(d["features"])
        result.append(d)
    return result


# ─── Strategy performance snapshot ───────────────────────────────────────────

def upsert_strategy_performance(strategy_name: str, capital: float,
                                 daily_pnl: float, cumulative_pnl: float,
                                 win_rate: float, total_trades: int,
                                 open_positions: int):
    conn = get_conn()
    today = utc_now().strftime("%Y-%m-%d")
    conn.execute("""
        INSERT INTO strategy_performance
        (strategy_name, date, capital, daily_pnl, cumulative_pnl, win_rate,
         total_trades, open_positions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy_name, date) DO UPDATE SET
            capital = excluded.capital,
            daily_pnl = excluded.daily_pnl,
            cumulative_pnl = excluded.cumulative_pnl,
            win_rate = excluded.win_rate,
            total_trades = excluded.total_trades,
            open_positions = excluded.open_positions
    """, (strategy_name, today, capital, daily_pnl, cumulative_pnl,
          win_rate, total_trades, open_positions))
    conn.commit()


def get_strategy_performance_history(strategy_name: str, days: int = 90) -> List[Dict]:
    rows = get_conn().execute("""
        SELECT * FROM strategy_performance
        WHERE strategy_name=?
          AND date >= date('now', ? || ' days')
        ORDER BY date ASC
    """, (strategy_name, f"-{days}")).fetchall()
    return [dict(row) for row in rows]
