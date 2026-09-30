"""
Daily report (Phase 4)
──────────────────────
Once per UTC day, shortly after midnight, the bot writes
DATA_DIR/reports/informe_diario_YYYY-MM-DD.txt about the day that just closed:

  • results of the learning book (on the risk budget), lab and baseline, in %,
    next to buy & hold of BTC and of an equal-weight basket of the traded coins
  • results per coin
  • strategies the evaluator ADDED (now allowed to trade) or REMOVED
    (discarded = value 0 for good, or back to test)
  • parameter changes of the learning (applied / rolled back)
  • risk events: kill switch, entries blocked by the risk engine or the evaluator
  • a market snapshot per coin (top traders, Hyperliquid whales, CME, funding)

It only reports facts from the database; it gives no advice. The daily review
(Claude) reads it and writes the feedback. Paper trading: no real money.
"""

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Optional

import pandas as pd

import config
import database as db
from utils import coin_of

logger = logging.getLogger(__name__)

ALLOWED = ("VIABLE", "CONDICIONAL")


def _day_bounds(day: date):
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return start.isoformat(), (start + timedelta(days=1)).isoformat()


class DailyReporter:

    def __init__(self, out_dir: str, books: Dict[str, tuple],
                 prices_fn: Callable[[], Dict[str, float]],
                 closes_fn: Callable[[str], pd.Series],
                 clock: Optional[Callable[[], datetime]] = None):
        """
        books     : label -> (book name, PortfolioManager); 'aprende' is the learning book
        prices_fn : current {symbol: price}
        closes_fn : symbol -> daily closes (index = candle open time, UTC)
        """
        self.out_dir = out_dir
        self.books = books
        self.prices_fn = prices_fn
        self.closes_fn = closes_fn
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    # ─── Schedule ────────────────────────────────────────────────────────────

    def write_if_due(self) -> Optional[str]:
        """Write the report of the previous UTC day once; returns its path."""
        day = self.clock().date() - timedelta(days=1)
        if db.get_meta("daily_report:last_day") == day.isoformat():
            return None
        text = self.build(day, remember=True)
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"informe_diario_{day.isoformat()}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        db.set_meta("daily_report:last_day", day.isoformat())
        db.set_meta("daily_report:latest", path)
        logger.info(f"[report] daily report written: {path}")
        return path

    # ─── Report ──────────────────────────────────────────────────────────────

    def build(self, day: date, remember: bool = False) -> str:
        t0, t1 = _day_bounds(day)
        symbols = list(config.SYMBOLS)
        out = []
        w = out.append
        w(f"INFORME DIARIO {day.isoformat()} (UTC) — PAPER TRADING, SIN DINERO REAL")
        w("=" * 72)
        w(f"Generado: {self.clock():%Y-%m-%d %H:%M} UTC | criptos: {', '.join(coin_of(s) for s in symbols)}")
        try:
            from risk_engine import RiskSettings, profile
            s = RiskSettings.load()
            p = profile(s.aggressiveness)
            w(f"Fondos ${s.funds:,.0f} | presupuesto ${s.budget:,.0f} | agresividad {p['level']}/10 "
              f"(opera: {', '.join(p['statuses'])}) | modo {s.mode}")
        except Exception:
            p = {"statuses": list(ALLOWED)}
        w("")
        self._results(w, day, symbols, remember)
        self._per_coin(w, t0, t1, symbols)
        self._strategies(w, t0, t1, p["statuses"])
        self._confirmations(w, t0, t1)
        self._learning(w, t0, t1)
        self._risk(w, t0, t1)
        self._market(w, symbols)
        w("Recordatorio: es una simulacion (paper trading). Un dia es una muestra minima;")
        w("ningun resultado diario prueba ni descarta que una estrategia funcione.")
        return "\n".join(out)

    def _results(self, w, day, symbols, remember):
        prices = self.prices_fn() or {}
        w("RESULTADOS (equity actual y cambio; % sobre el capital de cada libro)")
        for label, (book, pm) in self.books.items():
            base = pm.capital_base if pm.capital_base is not None else config.INITIAL_CAPITAL
            equity = sum(pm.strategy_equity(n, prices) for n in pm.strategies)
            prev = db.get_meta(f"daily_report:equity:{label}")
            if prev is None:
                change = f"sin informe anterior; total {(equity - base) / base:+.2%} desde el inicio"
            else:
                change = f"{(equity - float(prev)) / base:+.2%} vs informe anterior"
            trades = self._trades(book, *_day_bounds(day))
            pnl = sum(float(t["pnl"]) for t in trades)
            wins = sum(float(t["pnl"]) > 0 for t in trades)
            w(f"  {label:<9} equity ${equity:,.2f} (capital ${base:,.0f}) | {change} | "
              f"trades cerrados {len(trades)} (con ganancia {wins}), P&L ${pnl:+,.2f}")
            if remember:
                db.set_meta(f"daily_report:equity:{label}", repr(equity))
        rets = {}
        for sym in symbols:
            r = self._day_return(sym, day)
            if r is not None:
                rets[sym] = r
        if config.SYMBOL in rets:
            w(f"  BTC buy & hold del dia: {rets[config.SYMBOL]:+.2%}")
        else:
            w("  BTC buy & hold del dia: sin datos")
        if rets:
            w(f"  Canasta buy & hold (mismo peso): {sum(rets.values()) / len(rets):+.2%} ("
              + ", ".join(f"{coin_of(s)} {r:+.1%}" for s, r in rets.items()) + ")")
        w("")

    def _day_return(self, symbol, day) -> Optional[float]:
        try:
            closes = self.closes_fn(symbol)
            if closes is None or len(closes) < 2:
                return None
            idx = pd.Timestamp(day, tz="UTC")
            if idx not in closes.index:
                return None
            pos = closes.index.get_loc(idx)
            return float(closes.iloc[pos]) / float(closes.iloc[pos - 1]) - 1 if pos > 0 else None
        except Exception as e:
            logger.debug(f"[report] {symbol} closes: {e}")
            return None

    @staticmethod
    def _trades(book, t0, t1):
        return [t for t in db.get_trades(limit=10**7, book=book) if t0 <= (t["closed_at"] or "") < t1]

    def _per_coin(self, w, t0, t1, symbols):
        w("POR CRIPTO (P&L de trades cerrados en el dia, USD)")
        w(f"{'Cripto':<7}" + "".join(f"{label:>12}" for label in self.books) + f"{'trades aprende':>16}")
        learn_book = self.books.get("aprende", (None,))[0]
        by = {label: self._trades(book, t0, t1) for label, (book, _) in self.books.items()}
        for sym in symbols:
            row = f"{coin_of(sym):<7}"
            for label in self.books:
                row += f"{sum(float(t['pnl']) for t in by[label] if t['symbol'] == sym):>+12,.2f}"
            n = sum(t["symbol"] == sym for t in self._trades(learn_book, t0, t1)) if learn_book else 0
            w(row + f"{n:>16d}")
        w("")

    def _strategies(self, w, t0, t1, allowed_now):
        # Net change of the day: status when the day started vs when it ended
        start, end = {}, {}
        for e in sorted(db.get_strategy_evaluations(limit=10**7), key=lambda e: e["ts"]):
            if e["ts"] < t0:
                start[e["strategy_name"]] = e
            if e["ts"] < t1:
                end[e["strategy_name"]] = e
        added, removed = [], []
        for name, e in sorted(end.items()):
            before = start[name]["status"] if name in start else None
            if before == e["status"]:
                continue
            line = f"  {name:<28} {(before or 'nueva') + ' -> ' + e['status']:<26} {(e['reason'] or '')[:60]}"
            if e["status"] in ALLOWED:
                added.append(line)
            elif e["status"] == "DESCARTADA":
                removed.append(line + " (valor 0, no se repite)")
            elif before in ALLOWED:
                removed.append(line + " (vuelve a prueba)")
        w("ESTRATEGIAS AGREGADAS (el evaluador ahora las deja operar)")
        out = added or ["  ninguna"]
        for line in out:
            w(line)
        w("ESTRATEGIAS QUITADAS (descartadas o suspendidas)")
        for line in removed or ["  ninguna"]:
            w(line)
        statuses = db.get_all_strategy_status()
        counts = {k: sum(s["status"] == k for s in statuses)
                  for k in ("VIABLE", "CONDICIONAL", "EN_PRUEBA", "DESCARTADA")}
        w("Estado actual: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f" | con esta agresividad operan: {', '.join(allowed_now)}")
        w("")

    def _confirmations(self, w, t0, t1):
        """Blocked signals vs what the lab (no filter) did with the same signal."""
        learn_book = self.books.get("aprende", ("main",))[0]
        rows = db.get_conn().execute(
            "SELECT strategy_name, reason, recorded_at FROM signal_log WHERE book=? AND acted=0 "
            "AND reason LIKE 'confirmaciones:%' AND recorded_at >= ? AND recorded_at < ?",
            (learn_book, t0, t1)).fetchall()
        lab = db.get_trades(limit=10**7, book="lab")
        lost = won = pending = 0
        for r in rows:
            lo = (datetime.fromisoformat(r["recorded_at"]) - timedelta(minutes=5)).isoformat()
            hi = (datetime.fromisoformat(r["recorded_at"]) + timedelta(minutes=15)).isoformat()
            twin = [t for t in lab if t["strategy_name"] == r["strategy_name"] and lo <= t["entry_time"] <= hi]
            if twin:
                won += float(twin[0]["pnl"]) > 0
                lost += float(twin[0]["pnl"]) <= 0
            else:
                pending += 1      # still open in the lab, or the lab did not take it
        vetoes = sum("VETO" in (r["reason"] or "") for r in rows)
        w("CONFIRMACIONES (libro que aprende)")
        w(f"  {len(rows)} senales bloqueadas por confirmaciones ({vetoes} por veto, "
          f"{len(rows) - vetoes} por faltar confirmaciones)")
        if rows:
            plural = lambda n, one, many: f"{n} {one if n == 1 else many}"
            w(f"  El lab (sin filtro) con esas mismas senales: {plural(lost, 'habria perdido', 'habrian perdido')}, "
              f"{plural(won, 'habria ganado', 'habrian ganado')}, {pending} sin resultado todavia")
        groups = {">= 3": [], "<= 2": []}
        for t in self._trades(learn_book, t0, t1):
            feats = t.get("entry_features") or {}
            if isinstance(feats, str):
                try:
                    feats = json.loads(feats)
                except ValueError:
                    feats = {}
            net = (feats.get("confirmaciones") or {}).get("neto")
            if net is not None:
                groups[">= 3" if net >= 3 else "<= 2"].append(float(t["pnl"]))
        w("  Trades cerrados por confirmaciones: " + " | ".join(
            f"neto {k}: {len(v)} trades, {sum(p > 0 for p in v)} con ganancia" for k, v in groups.items()))
        w("")

    def _learning(self, w, t0, t1):
        audit = [r for r in db.get_learning_audit(limit=10**6) if t0 <= r["ts"] < t1]
        done = [r for r in audit if r["decision"] in ("applied", "rollback")]
        w(f"APRENDIZAJE: {len(audit)} propuestas en el dia, "
          f"{sum(r['decision'] == 'applied' for r in audit)} aplicadas, "
          f"{sum(r['decision'] == 'rollback' for r in audit)} revertidas")
        for r in sorted(done, key=lambda r: r["ts"]):
            w(f"  {r['ts'][11:16]} {r['decision']:<8} {r['strategy_name']} {r['param']} "
              f"{r['old_value']:g} -> {r['new_value']:g}  {(r['reason'] or '')[:60]}")
        w("")

    def _risk(self, w, t0, t1):
        learn_book = self.books.get("aprende", ("main",))[0]
        w("RIESGO (libro que aprende)")
        kill = json.loads(db.get_meta(f"risk:{learn_book}:kill") or "null")
        w(f"  Freno de emergencia: {'ACTIVO desde ' + kill['at'][:16] + ' - ' + kill['reason'] if kill else 'no'}")
        rows = db.get_conn().execute(
            "SELECT reason, COUNT(*) AS n FROM signal_log WHERE book=? AND acted=0 "
            "AND recorded_at >= ? AND recorded_at < ? GROUP BY reason", (learn_book, t0, t1)).fetchall()
        risk = {}
        evaluator = 0
        for r in rows:
            reason = r["reason"] or ""
            if reason.startswith("riesgo:"):
                key = reason.split("(")[0].strip()
                risk[key] = risk.get(key, 0) + r["n"]
            elif reason.startswith("evaluator"):
                evaluator += r["n"]
        w(f"  Senales bloqueadas por el evaluador: {evaluator}")
        if risk:
            w("  Entradas bloqueadas por el motor de riesgo:")
            for reason, n in sorted(risk.items(), key=lambda x: -x[1]):
                w(f"    {n} x {reason}")
        else:
            w("  Entradas bloqueadas por el motor de riesgo: 0")
        w("")

    def _market(self, w, symbols):
        from market_data import load_series

        def last(sym, metric):
            s = load_series(sym, metric)
            return float(s.iloc[-1]) if len(s) else None

        w("MERCADO (ultimo dato de cada fuente)")
        fng = last(config.SYMBOL, "fng")
        w(f"  Miedo y Codicia: {fng:.0f}" if fng is not None else "  Miedo y Codicia: sin datos")
        pct = lambda r: f"{r / (1 + r) * 100:.0f}% largo" if r is not None else "-"
        for sym in symbols:
            parts = [f"top Binance {pct(last(sym, 'top_pos_ratio'))}",
                     f"top OKX {pct(last(sym, 'okx_top_pos_ratio'))}"]
            hl = last(sym, "hl_top_net")
            parts.append(f"ballenas HL {hl:+.0%}" if hl is not None else "ballenas HL -")
            fr = last(sym, "funding_rate")
            parts.append(f"funding {fr * 100:+.4f}%" if fr is not None else "funding -")
            am = last(sym, "cot_am_net")
            if am is not None:
                parts.append(f"CME instituciones {am:+.1%}")
            w(f"  {coin_of(sym):<5} " + " | ".join(parts))
        w("")
