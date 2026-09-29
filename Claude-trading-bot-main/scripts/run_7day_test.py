"""
7-day paper-trading test run (runs inside the container instead of main.py)
──────────────────────────────────────────────────────────────────────────
  • Start and planned end are stored in the DB. After a container/server
    restart the run RESUMES (same schedule, no reset of equity or history;
    positions, trades, learned params and the trading mode come back from the
    DB) and the downtime is recorded as an interruption (from / to).
  • Every TEST_SNAPSHOT_MINUTES (60): one snapshot row per book (learning book
    and frozen baseline) and strategy + a TOTAL row: equity, free capital, open
    positions, closed trades, realized / unrealized PnL, drawdown, learning
    decisions so far.
  • Once per day: cumulative CSV export (snapshots + closed trades).
  • At the planned end: final snapshot, CSV and REPORTE_7DIAS.txt, trading
    stops. The container then only serves the dashboard (or exits with
    TEST_EXIT_WHEN_DONE=true, used by the short 30-minute trial).

Environment:
  TEST_NAME (7d) · TEST_DURATION_HOURS (168) · TEST_SNAPSHOT_MINUTES (60)
  TEST_EXIT_WHEN_DONE (false) · TEST_REPORT_NAME (REPORTE_7DIAS.txt)
Outputs go to DATA_DIR/reports/.
"""

import csv
import logging
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database as db

logger = logging.getLogger("test_run")

# Success criteria (Part 5.4) — all must hold, over the learning book.
MIN_TRADES = 20
MAX_DRAWDOWN = 0.10
MIN_PROFIT_FACTOR = 1.2


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat()


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


class TestRun:

    def __init__(self, name: str, duration_hours: float, snapshot_minutes: float,
                 out_dir: str, report_name: str = "REPORTE_7DIAS.txt"):
        self.name = name
        self.duration = timedelta(hours=duration_hours)
        self.snapshot_every = timedelta(minutes=snapshot_minutes)
        self.out_dir = out_dir
        self.report_name = report_name
        os.makedirs(out_dir, exist_ok=True)
        conn = db.get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL, ts TEXT NOT NULL, book TEXT NOT NULL,
                strategy_name TEXT NOT NULL, price REAL, equity REAL, free_capital REAL,
                open_positions INTEGER, closed_trades INTEGER, realized_pnl REAL,
                unrealized_pnl REAL, drawdown_pct REAL, learning_applied INTEGER,
                learning_rejected INTEGER, learning_rollback INTEGER
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_interruptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL, down_from TEXT NOT NULL, down_to TEXT NOT NULL,
                seconds REAL NOT NULL
            )""")
        conn.commit()

    # ─── Schedule ────────────────────────────────────────────────────────────

    def _key(self, suffix: str) -> str:
        return f"test:{self.name}:{suffix}"

    @property
    def start(self) -> datetime:
        return _parse(db.get_meta(self._key("start")))

    @property
    def end(self) -> datetime:
        return _parse(db.get_meta(self._key("end")))

    @property
    def finished(self) -> bool:
        return db.get_meta(self._key("finished")) is not None

    def start_or_resume(self, now: datetime) -> str:
        if self.finished:
            return "finished"
        if db.get_meta(self._key("start")) is None:
            if db.get_trades(limit=1, book=None) or db.get_open_positions(book=None):
                logger.warning(f"[test {self.name}] DATA_DIR already has trades/positions: "
                               f"results assume a fresh start with ${config.INITIAL_CAPITAL:,.0f}")
            db.set_meta(self._key("start"), _iso(now))
            db.set_meta(self._key("end"), _iso(now + self.duration))
            self.heartbeat(now)
            logger.info(f"[test {self.name}] STARTED {now:%Y-%m-%d %H:%M} UTC, "
                        f"ends {now + self.duration:%Y-%m-%d %H:%M} UTC")
            return "started"
        last = db.get_meta(self._key("heartbeat"))
        down_from = _parse(last) if last else now
        conn = db.get_conn()
        conn.execute("INSERT INTO test_interruptions (test_name, down_from, down_to, seconds) "
                     "VALUES (?, ?, ?, ?)", (self.name, _iso(down_from), _iso(now),
                                             (now - down_from).total_seconds()))
        conn.commit()
        self.heartbeat(now)
        logger.warning(f"[test {self.name}] RESUMED after interruption "
                       f"{down_from:%Y-%m-%d %H:%M} -> {now:%H:%M} UTC; ends {self.end:%Y-%m-%d %H:%M} UTC")
        return "resumed"

    def heartbeat(self, now: datetime):
        db.set_meta(self._key("heartbeat"), _iso(now))

    def is_over(self, now: datetime) -> bool:
        return now >= self.end

    def interruptions(self) -> list:
        rows = db.get_conn().execute(
            "SELECT * FROM test_interruptions WHERE test_name=? ORDER BY id", (self.name,)).fetchall()
        return [dict(r) for r in rows]

    # ─── Snapshots ───────────────────────────────────────────────────────────

    def snapshot_due(self, now: datetime) -> bool:
        last = db.get_meta(self._key("last_snapshot"))
        return last is None or now - _parse(last) >= self.snapshot_every

    def snapshot(self, now: datetime, bot):
        price = bot._current_price
        since = _iso(self.start)
        audit = [r for r in db.get_learning_audit(limit=10**6) if r["ts"] >= since]
        learning = {d: sum(r["decision"] == d for r in audit) for d in ("applied", "rejected", "rollback")}
        conn = db.get_conn()
        for book, pm, strats in ((bot.book, bot.portfolio, bot.strategies),
                                 ("baseline", bot.baseline_portfolio, bot.baselines)):
            totals = dict(equity=0.0, free=0.0, open=0, closed=0, realized=0.0, unreal=0.0)
            for s in strats:
                if not s.is_active:
                    continue
                positions = db.get_open_positions(s.name, book=book)
                unreal = sum(((price - p["entry_price"]) if p["side"] == "LONG"
                              else (p["entry_price"] - price)) * p["quantity"] for p in positions)
                trades = [t for t in db.get_trades(s.name, limit=10**6, book=book)
                          if (t["closed_at"] or "") >= since]
                row = dict(equity=pm.strategy_equity(s.name, price), free=pm._capital.get(s.name, 0.0),
                           open=len(positions), closed=len(trades),
                           realized=sum(t["pnl"] for t in trades), unreal=unreal)
                self._insert(conn, now, book, s.name, price, row, learning)
                for k in totals:
                    totals[k] += row[k]
            self._insert(conn, now, book, "TOTAL", price, totals, learning)
        conn.commit()
        db.set_meta(self._key("last_snapshot"), _iso(now))

    def _insert(self, conn, now, book, name, price, row, learning):
        peak = conn.execute("SELECT MAX(equity) FROM test_snapshots WHERE test_name=? AND book=? "
                            "AND strategy_name=?", (self.name, book, name)).fetchone()[0]
        peak = max(peak or row["equity"], row["equity"])
        dd = (peak - row["equity"]) / peak if peak > 0 else 0.0
        conn.execute("""
            INSERT INTO test_snapshots (test_name, ts, book, strategy_name, price, equity,
                free_capital, open_positions, closed_trades, realized_pnl, unrealized_pnl,
                drawdown_pct, learning_applied, learning_rejected, learning_rollback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.name, _iso(now), book, name, price, row["equity"], row["free"], row["open"],
              row["closed"], row["realized"], row["unreal"], dd,
              learning["applied"], learning["rejected"], learning["rollback"]))

    def snapshots(self) -> list:
        rows = db.get_conn().execute(
            "SELECT * FROM test_snapshots WHERE test_name=? ORDER BY id", (self.name,)).fetchall()
        return [dict(r) for r in rows]

    # ─── Daily CSV ───────────────────────────────────────────────────────────

    def export_daily(self, now: datetime, force: bool = False) -> list:
        day = int((now - self.start) / timedelta(days=1))
        done = int(db.get_meta(self._key("exported_day")) or 0)
        if not force and day <= done:
            return []
        db.set_meta(self._key("exported_day"), str(max(day, done)))
        tag = f"dia{day}" if not force else "final"
        paths = []
        snaps = self.snapshots()
        if snaps:
            paths.append(self._write_csv(f"test_{self.name}_snapshots_{tag}.csv", snaps))
        since = _iso(self.start)
        trades = [t for t in db.get_trades(limit=10**7, book=None) if (t["closed_at"] or "") >= since]
        if trades:
            paths.append(self._write_csv(f"test_{self.name}_trades_{tag}.csv", trades))
        return paths

    def _write_csv(self, filename: str, rows: list) -> str:
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
        return path

    # ─── End of test ─────────────────────────────────────────────────────────

    def book_metrics(self, book: str) -> dict:
        totals = [r for r in self.snapshots() if r["book"] == book and r["strategy_name"] == "TOTAL"]
        since = _iso(self.start)
        trades = [t for t in db.get_trades(limit=10**7, book=book) if (t["closed_at"] or "") >= since]
        pnls = [float(t["pnl"]) for t in trades]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = -sum(p for p in pnls if p <= 0)
        # Each book starts with INITIAL_CAPITAL (the run needs a fresh DATA_DIR); the
        # first snapshot can already include a position opened at startup.
        equity_start = config.INITIAL_CAPITAL
        equity_end = totals[-1]["equity"] if totals else equity_start
        return dict(
            equity_start=equity_start, equity_end=equity_end,
            total_return=equity_end / equity_start - 1 if equity_start else 0.0,
            max_drawdown=max((r["drawdown_pct"] for r in totals), default=0.0),
            trades=len(pnls),
            profit_factor=(gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0),
            win_rate=(sum(p > 0 for p in pnls) / len(pnls)) if pnls else 0.0,
            fees=sum(float(t["fees_paid"]) for t in trades),
            pnl_pcts=[float(t["pnl_pct"]) for t in trades],
        )

    def finish(self, now: datetime, bot) -> str:
        self.snapshot(now, bot)
        self.export_daily(now, force=True)
        path = os.path.join(self.out_dir, self.report_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.report(now, bot.book) + "\n")
        db.set_meta(self._key("finished"), _iso(now))
        logger.warning(f"[test {self.name}] FINISHED — report: {path}")
        return path

    def report(self, now: datetime, learn_book: str) -> str:
        L, B = self.book_metrics(learn_book), self.book_metrics("baseline")
        gaps = self.interruptions()
        down = sum(g["seconds"] for g in gaps)
        label, checks = verdict(L, B)
        out = []
        w = out.append
        w(f"REPORTE PRUEBA '{self.name}' — PAPER TRADING BTCUSDT (demo, sin dinero real)")
        w("=" * 74)
        w(f"Inicio  : {self.start:%Y-%m-%d %H:%M} UTC")
        w(f"Fin     : {now:%Y-%m-%d %H:%M} UTC (planificado {self.end:%Y-%m-%d %H:%M})")
        w(f"Modo    : {db.get_meta('trading_mode') or '?'} (libro que aprende: '{learn_book}')")
        w(f"Interrupciones: {len(gaps)} (total {down / 3600:.2f} h sin operar)")
        for g in gaps:
            w(f"  - {g['down_from'][:16]} -> {g['down_to'][:16]}  ({g['seconds'] / 60:.1f} min)")
        w("")
        rows = [("Equity inicial", "${:,.2f}", "equity_start"), ("Equity final", "${:,.2f}", "equity_end"),
                ("Retorno (neto de fees)", "{:+.2%}", "total_return"),
                ("Max drawdown (snapshots)", "{:.2%}", "max_drawdown"),
                ("Trades cerrados", "{:d}", "trades"), ("Profit factor", "{:.2f}", "profit_factor"),
                ("Win rate", "{:.1%}", "win_rate"), ("Fees pagadas", "${:,.2f}", "fees")]
        w(f"{'Metrica':<28}{'APRENDE':>16}{'BASELINE':>16}")
        w("-" * 60)
        for name, fmt, key in rows:
            w(f"{name:<28}{fmt.format(L[key]):>16}{fmt.format(B[key]):>16}")
        w(f"Aprende - baseline (equity final): ${L['equity_end'] - B['equity_end']:+,.2f}")
        w("")
        final = {}
        for r in self.snapshots():
            if r["strategy_name"] != "TOTAL":
                final[(r["book"], r["strategy_name"])] = r["equity"]
        names = sorted({n for _, n in final})
        if names:
            w(f"{'Estrategia':<24}{'Aprende':>12}{'Baseline':>12}")
            for n in names:
                w(f"{n:<24}{final.get((learn_book, n), 0):>12,.2f}{final.get(('baseline', n), 0):>12,.2f}")
            w("")
        since = _iso(self.start)
        audit = [r for r in db.get_learning_audit(limit=10**6) if r["ts"] >= since]
        w(f"Aprendizaje durante la prueba: {len(audit)} propuestas, "
          f"{sum(r['decision'] == 'applied' for r in audit)} aplicadas, "
          f"{sum(r['decision'] == 'rollback' for r in audit)} revertidas")
        for r in reversed(audit):
            if r["decision"] in ("applied", "rollback"):
                w(f"  {r['ts'][:16]} {r['decision']:<8} {r['strategy_name']} {r['param']} "
                  f"{r['old_value']:g}->{r['new_value']:g}  {r['reason'][:60]}")
        w("")
        w("Criterios de exito (sobre el libro que aprende):")
        for ok, text in checks:
            w(f"  [{'OK' if ok else 'NO'}] {text}")
        w(f"Veredicto: {label}")
        w("")
        w("Significancia estadistica:")
        w("  " + significance(L["pnl_pcts"]))
        w("  Aprende vs baseline: con pocos dias y pocos trades no hay forma de")
        w("  distinguir la diferencia del azar; no se reporta como significativa.")
        w("")
        w("Conclusion honesta: 7 dias con estrategias de velas 1d/4h son una muestra muy")
        w("chica. Un buen resultado NO prueba rentabilidad y uno malo NO la descarta.")
        w("Es paper trading con fills simulados (sin spread real, latencia ni ejecucion).")
        return "\n".join(out)


def verdict(learn: dict, base: dict):
    """(label, [(ok, description), ...]) for the Part 5.4 success criteria."""
    checks = [
        (learn["equity_end"] > learn["equity_start"], "equity final > inicial (neto de fees)"),
        (learn["max_drawdown"] < MAX_DRAWDOWN, f"max drawdown < {MAX_DRAWDOWN:.0%}"),
        (learn["trades"] >= MIN_TRADES, f"al menos {MIN_TRADES} trades cerrados"),
        (learn["profit_factor"] > MIN_PROFIT_FACTOR, f"profit factor > {MIN_PROFIT_FACTOR}"),
        (learn["equity_end"] > base["equity_end"], "la version que aprende supera al baseline"),
    ]
    if learn["trades"] < MIN_TRADES:
        return f"NO CONCLUYENTE (solo {learn['trades']} trades; muestra insuficiente)", checks
    if learn["equity_end"] <= learn["equity_start"] or learn["profit_factor"] < 1:
        return "NO RENTABLE", checks
    if all(ok for ok, _ in checks):
        return "CUMPLE LOS CRITERIOS (no prueba rentabilidad futura)", checks
    return "NO CONCLUYENTE (no cumple todos los criterios)", checks


def significance(pnl_pcts: list) -> str:
    n = len(pnl_pcts)
    if n < 2:
        return f"{n} trades: imposible evaluar significancia."
    try:
        from scipy import stats
        t, p = stats.ttest_1samp(pnl_pcts, 0.0, alternative="greater")
        return (f"{n} trades; t-test 'PnL medio por trade > 0': p = {p:.3f} "
                f"({'significativo al 5%' if p < 0.05 else 'NO significativo'}). "
                f"{'Con menos de 30 trades el test es poco fiable.' if n < 30 else ''}")
    except Exception as e:
        return f"{n} trades; no se pudo calcular el test ({e})."


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    import main as bot_main

    name = os.getenv("TEST_NAME", "7d")
    run_kwargs = dict(
        name=name,
        duration_hours=float(os.getenv("TEST_DURATION_HOURS", "168")),
        snapshot_minutes=float(os.getenv("TEST_SNAPSHOT_MINUTES", "60")),
        out_dir=os.path.join(config.DATA_DIR, "reports"),
        report_name=os.getenv("TEST_REPORT_NAME", "REPORTE_7DIAS.txt"),
    )
    exit_when_done = os.getenv("TEST_EXIT_WHEN_DONE", "false").lower() == "true"
    now = lambda: datetime.now(timezone.utc)

    bot_main.prepare_database()
    run = TestRun(**run_kwargs)
    state = run.start_or_resume(now())
    if state == "finished":
        logger.warning(f"[test {name}] already finished — not trading")
        if not exit_when_done:
            signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
            from dashboard.app import run_dashboard
            run_dashboard(debug=False)   # read-only review of the results
        return

    bot = bot_main.TradingBot()
    bot.lock_mode = True

    def monitor():
        while not bot_main._shutdown.is_set():
            try:
                t = now()
                run.heartbeat(t)
                if bot.portfolio is not None and bot.baseline_portfolio is not None:
                    if run.snapshot_due(t):
                        run.snapshot(t, bot)
                    for p in run.export_daily(t):
                        logger.info(f"[test {name}] daily CSV: {p}")
                    if run.is_over(t):
                        run.finish(t, bot)
                        bot_main._shutdown.set()
                        return
            except Exception as e:
                logger.error(f"[test {name}] monitor error: {e}", exc_info=True)
            bot_main._shutdown.wait(30)

    threading.Thread(target=monitor, daemon=True, name="test-monitor").start()
    try:
        bot.run()          # returns once _shutdown is set (end of test or SIGTERM)
    except ConnectionError as e:
        logger.critical(f"Cannot connect to Binance: {e}")
        sys.exit(1)

    if run.finished and not exit_when_done:
        logger.warning(f"[test {name}] trading stopped; dashboard stays up for review")
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
