"""
Accelerated historical replay: "learning" vs frozen "baseline"
──────────────────────────────────────────────────────────────
Replays real Binance candles hour by hour through the SAME code the live bot
runs (main.TradingBot._process_signals / _journal_new_trades, PortfolioManager,
LearningEngine, AdaptiveTuner) with the clock set to historical time.

  • Every closed 1h candle: SL/TP are checked along an intra-hour price path
    open → low/high → high/low → close that passes through every SL/TP level
    (as a continuously checked live price would).
  • When a 4h / 1d candle closes: strategies of that interval get the last
    LOOKBACK_CANDLES closed candles and their signals are processed at the
    1h close price — the live loop does the same within a minute.
  • The tuner runs on replay time with only candles closed before "now".

Both books start with INITIAL_CAPITAL split over all 8 strategies and use
simulated fills (price ± SLIPPAGE, fee 0.1%) — like the live 'observe' and
'baseline' books. This is a replay/backtest, NOT live evidence.

Usage:
    python run_replay_test.py              # 500 days
    python run_replay_test.py --days 365
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import database as db

DELTA = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}
LEARN_BOOK = "observe"   # the learning strategies' book, as in live observation mode

logger = logging.getLogger("replay")


# ─── Intra-candle price path ──────────────────────────────────────────────────

def intrabar_path(o: float, h: float, l: float, c: float, levels) -> list:
    """
    Prices visited inside one candle: up candle open→low→high→close, down candle
    open→high→low→close. Every level (SL/TP) crossed by a leg is inserted in the
    order the price would reach it.
    """
    o, h, l, c = float(o), float(h), float(l), float(c)
    first, second = (l, h) if c >= o else (h, l)
    path = [o]
    for a, b in ((o, first), (first, second), (second, c)):
        inside = sorted((x for x in levels if min(a, b) < x < max(a, b)), reverse=b < a)
        path.extend(inside)
        path.append(b)
    return path


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_data(replay_days: int, cache_dir: str) -> dict:
    """Real Binance candles (closed only, with indicators), cached per day."""
    from binance_client import BINANCE_PUBLIC_BASE, BinancePublicDataFetcher
    history_days = replay_days + max(
        config.LEARNING_PROPOSAL_DAYS + config.LEARNING_VALIDATION_DAYS + config.LEARNING_WARMUP_DAYS,
        config.EVAL_WINDOWS * config.EVAL_WINDOW_DAYS + config.EVAL_WARMUP_DAYS) + 10
    os.makedirs(cache_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(cache_dir, f"klines_{config.SYMBOL}_{replay_days}d_{history_days}h_{stamp}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    fetcher = BinancePublicDataFetcher(BINANCE_PUBLIC_BASE)
    # 1h is both the replay clock and the interval of some catalog strategies
    data = {iv: fetcher.get_klines_since(config.SYMBOL, iv, history_days) for iv in ("1h", "4h", "1d")}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


# ─── Replay core ──────────────────────────────────────────────────────────────

def run_replay(data: dict, start: pd.Timestamp, end: pd.Timestamp, db_path: str,
               learning: bool = True, progress=None, eval_interval_hours: float = None,
               funds: float = None, budget: float = None, aggressiveness: int = None) -> dict:
    import main
    from adaptive_tuner import AdaptiveTuner
    from learning_engine import LearningEngine
    from risk_engine import RiskEngine, RiskSettings
    from strategies import ALL_STRATEGIES, CANDIDATE_STRATEGIES
    from strategy_evaluator import StrategyEvaluator

    config.DB_PATH = db_path
    conn = getattr(db._local, "conn", None)
    if conn is not None:
        conn.close()
    db._local.conn = None
    db.init_db()
    if eval_interval_hours:
        config.EVAL_INTERVAL_HOURS = eval_interval_hours
    if funds:
        config.INITIAL_CAPITAL = float(funds)     # lab / baseline books start with the funds
    settings = RiskSettings(funds=config.INITIAL_CAPITAL,
                            budget=float(budget or config.INITIAL_CAPITAL * 0.10),
                            aggressiveness=int(aggressiveness or config.RISK_AGGRESSIVENESS))
    settings.save()

    clock = {"now": start}
    iso_clock = lambda: clock["now"].isoformat()

    # Same line-up as the live bot: registered strategies + candidate catalog.
    learners = [S() for S in ALL_STRATEGIES + CANDIDATE_STRATEGIES]
    for s in learners:
        s.is_active = True
    registered = {S().name for S in ALL_STRATEGIES}
    baselines = main.build_baselines(registered)
    lab = main.build_lab(learners)

    # The live TradingBot without its network client: same methods, historical clock.
    bot = main.TradingBot.__new__(main.TradingBot)
    bot.strategies, bot.baselines, bot.lab = learners, baselines, lab
    bot.book, bot.mode = LEARN_BOOK, "OBSERVE"
    bot._strat_dfs, bot._logged_candle, bot._current_price = {}, {}, 0.0
    bot.risk = RiskEngine(LEARN_BOOK, clock=lambda: clock["now"])
    bot.portfolio = main.make_portfolio(None, learners, LEARN_BOOK, clock=iso_clock,
                                        capital_base=settings.budget, risk_engine=bot.risk)
    bot.baseline_portfolio = main.make_portfolio(None, baselines, "baseline", clock=iso_clock)
    bot.lab_portfolio = main.make_portfolio(None, lab, "lab", clock=iso_clock)
    bot.learning = LearningEngine({s.name: s for s in learners})

    def history(interval, days, now):
        df = data[interval]
        closed = df.iloc[:(df.index + DELTA[interval]).searchsorted(now, side="right")]
        return closed[closed.index >= now - pd.Timedelta(days=days)]

    bot.tuner = AdaptiveTuner(
        learners={s.name: s for s in learners if s.name in registered}, history_fn=history,
        equity_fn=bot._strategy_equity, clock=lambda: clock["now"], learner_book=LEARN_BOOK,
    )
    bot.evaluator = StrategyEvaluator({s.name: s for s in learners}, history_fn=history,
                                      clock=lambda: clock["now"], live_book="lab")

    intervals = sorted({s.candle_interval for s in learners})
    close_times = {iv: data[iv].index + DELTA[iv] for iv in intervals}
    last_closed = {iv: -1 for iv in close_times}
    hours = data["1h"][(data["1h"].index >= start) & (data["1h"].index < end)]
    equity = {"learn": [], "lab": [], "baseline": []}
    books = (("learn", bot.portfolio, learners), ("lab", bot.lab_portfolio, lab),
             ("baseline", bot.baseline_portfolio, baselines))
    t0 = time.time()

    for i, (ts, row) in enumerate(hours.iterrows()):
        # 1. SL/TP inside the hour
        clock["now"] = ts + pd.Timedelta(minutes=30)
        levels = [lvl for p in db.get_open_positions(book=None)
                  for lvl in (p["stop_loss"], p["take_profit"]) if lvl]
        for price in intrabar_path(row["open"], row["high"], row["low"], row["close"], levels):
            bot._current_price = price
            for _, pm, _ in books:
                pm.check_open_positions(price)

        # 2. Hour closed: new 4h / 1d candles → signals at the close price
        now, price = ts + DELTA["1h"], float(row["close"])
        clock["now"], bot._current_price = now, price
        bot._risk_housekeeping(price)          # day / kill-switch state, like the live loop
        changed = set()
        for iv, closes in close_times.items():
            n = closes.searchsorted(now, side="right")
            if n != last_closed[iv]:
                last_closed[iv] = n
                bot._strat_dfs[iv] = data[iv].iloc[max(0, n - config.LOOKBACK_CANDLES):n]
                changed.add(iv)
        if changed:
            fixed = lambda name, df: config.BASELINE_ML_CONFIDENCE
            bot._process_signals([s for s in learners if s.candle_interval in changed],
                                 bot.portfolio, price,
                                 lambda name, df: bot.learning.get_confidence(name, df),
                                 gate=bot._learner_gate)
            bot._process_signals([b for b in baselines if b.candle_interval in changed],
                                 bot.baseline_portfolio, price, fixed)
            bot._process_signals([s for s in lab if s.candle_interval in changed],
                                 bot.lab_portfolio, price, fixed, gate=bot._lab_gate)

        # 3. Journal + evaluator + learning, exactly as the live learning loop
        bot._journal_new_trades()
        if learning:
            bot.evaluator.run_cycle_if_due()
            bot.tuner.run_cycle_if_due()

        for label, pm, strats in books:
            equity[label].append(sum(pm.strategy_equity(s.name, price) for s in strats))
        if progress and i % (24 * 30) == 0:
            progress(ts, i, len(hours), time.time() - t0)

    final_price = float(hours["close"].iloc[-1])
    per_strategy = {
        label: {s.name: pm.strategy_equity(s.name, final_price) for s in strats}
        for label, pm, strats in books
    }
    idx = hours.index + DELTA["1h"]
    metrics = {
        "learn": _metrics(pd.Series(equity["learn"], index=idx), LEARN_BOOK, settings.budget),
        "lab": _metrics(pd.Series(equity["lab"], index=idx), "lab", config.INITIAL_CAPITAL),
        "baseline": _metrics(pd.Series(equity["baseline"], index=idx), "baseline", config.INITIAL_CAPITAL),
    }
    risk_events = {
        "kill_switch": db.get_meta(f"risk:{LEARN_BOOK}:kill"),
        "blocked": {r["reason"].split("(")[0].strip(): r["n"] for r in db.get_conn().execute(
            "SELECT reason, COUNT(*) AS n FROM signal_log WHERE book=? AND reason LIKE 'riesgo:%' "
            "GROUP BY reason", (LEARN_BOOK,)).fetchall()},
    }
    return {
        "metrics": metrics, "equity": equity, "index": idx, "per_strategy": per_strategy,
        "learners": learners, "baselines": baselines, "settings": settings, "risk_events": risk_events,
        "btc_return": final_price / float(hours["open"].iloc[0]) - 1,
        "start": start, "end": end, "elapsed_sec": time.time() - t0,
    }


def _metrics(eq: pd.Series, book: str, start_eq: float) -> dict:
    trades = db.get_trades(limit=10**7, book=book)
    pnls = [float(t["pnl"]) for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p <= 0)
    days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400, 1)
    total_ret = eq.iloc[-1] / start_eq - 1
    daily = eq.resample("1D").last().pct_change().dropna()
    return {
        "equity_start": start_eq,
        "equity_end": float(eq.iloc[-1]),
        "total_return": float(total_ret),
        "cagr": float((1 + total_ret) ** (365 / days) - 1),
        "max_drawdown": float(((eq.cummax() - eq) / eq.cummax()).max()),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("nan"),
        "win_rate": (sum(p > 0 for p in pnls) / len(pnls)) if pnls else 0.0,
        "trades": len(pnls),
        "fees": float(sum(float(t["fees_paid"]) for t in trades)),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0,
    }


# ─── Report ───────────────────────────────────────────────────────────────────

def _reason_category(reason: str) -> str:
    reason = reason or ""
    if "no neighbour" in reason:
        return "ningun vecino mejora la ventana de propuesta"
    if "trades" in reason and reason.startswith("validation"):
        return "validacion: pocos trades"
    if "profit factor" in reason:
        return "validacion: profit factor no mejora lo suficiente"
    if "drawdown" in reason:
        return "validacion: drawdown peor"
    if "history" in reason:
        return "historia insuficiente"
    return reason

def build_report(res: dict, data_source: str) -> str:
    L, T, B = res["metrics"]["learn"], res["metrics"]["lab"], res["metrics"]["baseline"]
    audit = db.get_learning_audit(limit=10**6)
    counts = {d: sum(r["decision"] == d for r in audit) for d in ("applied", "rejected", "rollback")}
    rows = [
        ("Retorno total", "{:+.2%}", "total_return"), ("CAGR", "{:+.2%}", "cagr"),
        ("Max drawdown", "{:.2%}", "max_drawdown"), ("Profit factor", "{:.2f}", "profit_factor"),
        ("Win rate", "{:.1%}", "win_rate"), ("Numero de trades", "{:d}", "trades"),
        ("Sharpe simple (diario, anualiz.)", "{:.2f}", "sharpe"),
        ("Equity final", "${:,.2f}", "equity_end"), ("Fees pagadas", "${:,.2f}", "fees"),
    ]
    out = []
    w = out.append
    w("REPLAY HISTORICO ACELERADO — APRENDE vs BASELINE")
    w("=" * 72)
    w(f"Generado        : {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    w(f"Periodo         : {res['start']:%Y-%m-%d %H:%M} -> {res['end']:%Y-%m-%d %H:%M} UTC "
      f"({(res['end'] - res['start']).days} dias)")
    w(f"Datos           : {data_source} (velas reales BTCUSDT 1h/4h/1d)")
    s = res["settings"]
    from risk_engine import profile
    p = profile(s.aggressiveness)
    w(f"Fondos (paper)  : ${s.funds:,.0f}  (capital de LAB y BASELINE)")
    w(f"Presupuesto     : ${s.budget:,.0f}  (capital de APRENDE)  |  agresividad {p['level']}/10: "
      f"riesgo/operacion {p['risk_per_trade']:.2%}, max {p['max_open']} posiciones, "
      f"limite diario {p['daily_loss']:.1%}, freno {p['max_drawdown']:.0%}, "
      f"opera: {', '.join(p['statuses'])}")
    w(f"BTC buy & hold  : {res['btc_return']:+.2%} en el mismo periodo (referencia)")
    w(f"Evaluador       : cada {config.EVAL_INTERVAL_HOURS:g} h de tiempo simulado "
      f"(en vivo: cada 24 h); tuner cada {config.LEARNING_INTERVAL_HOURS:g} h")
    w(f"Duracion calculo: {res['elapsed_sec'] / 60:.1f} min")
    w("")
    w("APRENDE  = 8 registradas + 9 del catalogo; solo opera lo que el evaluador aprueba")
    w("           para esta agresividad; tamano y limites del motor de riesgo sobre el")
    w("           PRESUPUESTO; parametros del tuner. Sus % son sobre el presupuesto.")
    w("LAB      = las mismas 17 sin filtro (menos las DESCARTADAS): la evidencia en vivo")
    w("BASELINE = las 8 originales con parametros por defecto congelados, sin filtro")
    w("")
    w(f"{'Metrica':<34}{'APRENDE':>15}{'LAB':>15}{'BASELINE':>15}")
    w("-" * 79)
    for label, fmt, key in rows:
        w(f"{label:<34}{fmt.format(L[key]):>15}{fmt.format(T[key]):>15}{fmt.format(B[key]):>15}")
    w("")
    diff = L["equity_end"] - B["equity_end"]
    better = diff > 0 and L["max_drawdown"] <= B["max_drawdown"] + 0.02
    w(f"Aprende vs baseline: equity {diff:+,.2f} USD -> {'SUPERO' if better else 'NO supero'} al baseline.")
    if B["trades"] and L["trades"] < 0.5 * B["trades"]:
        w(f"  OJO: hizo {L['trades']} trades vs {B['trades']} del baseline; buena parte de la")
        w("  diferencia viene de operar menos (el filtro evita estrategias no viables).")
    w("")
    ev = res["risk_events"]
    w("MOTOR DE RIESGO (libro que aprende)")
    kill = json.loads(ev["kill_switch"] or "null")
    w(f"  Freno de emergencia: {'ACTIVADO el ' + kill['at'][:10] + ' - ' + kill['reason'] if kill else 'no se activo'}")
    if ev["blocked"]:
        w("  Entradas bloqueadas por riesgo:")
        for reason, n in sorted(ev["blocked"].items(), key=lambda x: -x[1])[:8]:
            w(f"    {n:>5} x {reason}")
    w("")
    w("EQUITY FINAL POR ESTRATEGIA (aprende / lab / baseline) Y CALIFICACION FINAL")
    w(f"{'Estrategia':<22}{'Aprende':>10}{'Lab':>10}{'Baseline':>10}  {'Estado':<12}{'Punt.':>6}  Motivo")
    status = {s["strategy_name"]: s for s in db.get_all_strategy_status()}
    for name, eq_l in res["per_strategy"]["learn"].items():
        eq_t = res["per_strategy"]["lab"].get(name, 0.0)
        eq_b = res["per_strategy"]["baseline"].get(name)
        st = status.get(name, {})
        w(f"{name:<22}{eq_l:>10,.2f}{eq_t:>10,.2f}{(f'{eq_b:,.2f}' if eq_b is not None else '-'):>10}  "
          f"{st.get('status', '?'):<12}{st.get('score', 0):>6.2f}  {(st.get('reason') or '')[:70]}")
    evals = db.get_strategy_evaluations(limit=10**6)
    w(f"Evaluaciones realizadas: {len(evals)}")
    discarded = sorted((s for s in status.values() if s["status"] == "DESCARTADA"),
                       key=lambda s: s["discarded_at"] or "")
    for s in discarded:
        w(f"  descartada el {(s['discarded_at'] or '')[:10]}: {s['strategy_name']}")
    w("")
    w(f"APRENDIZAJE: {len(audit)} propuestas -> {counts['applied']} aplicadas, "
      f"{counts['rejected']} rechazadas, {counts['rollback']} revertidas (rollback)")
    reasons = pd.Series([_reason_category(r["reason"]) for r in audit if r["decision"] == "rejected"])
    if len(reasons):
        w("Motivos de rechazo mas frecuentes:")
        for reason, n in reasons.value_counts().head(6).items():
            w(f"  {n:>5} x {reason}")
    changes = [r for r in reversed(audit) if r["decision"] in ("applied", "rollback")]
    if changes:
        w("Cambios aplicados / revertidos (orden cronologico):")
        for r in changes:
            w(f"  {r['ts'][:10]}  {r['decision']:<8} {r['strategy_name']:<22} {r['param']:<17} "
              f"{r['old_value']:g} -> {r['new_value']:g}   {r['reason'][:70]}")
    w("Parametros finales (aprende) distintos del baseline:")
    any_diff = False
    for s in res["learners"]:
        d = {k: v for k, v in s.tunable_values().items() if v != type(s)().params[k]}
        if d:
            any_diff = True
            w(f"  {s.name}: {d}")
    if not any_diff:
        w("  (ninguno)")
    w("")
    w("ADVERTENCIAS (leer)")
    w("- Es un REPLAY/BACKTEST sobre datos pasados, no evidencia en vivo.")
    w("- Riesgo de sobreajuste: los parametros y umbrales se eligieron mirando datos")
    w("  historicos; un buen resultado aqui no garantiza nada hacia adelante.")
    w("- SL/TP se simulan con un recorrido intra-hora (apertura->min/max->cierre);")
    w("  el orden real dentro de la hora puede diferir. Fills simulados con")
    w(f"  slippage {config.SLIPPAGE:.2%} y fee {config.TRADING_FEE:.1%}; sin spread real ni latencia.")
    w("- El evaluador califica con backtests de los ~2 anos ANTERIORES a cada fecha")
    w("  simulada (nunca con datos futuros), pero los umbrales del evaluador y el")
    w("  catalogo se eligieron hoy: sigue habiendo riesgo de sesgo de seleccion.")
    w("- Aprende y lab reparten el capital entre 17 estrategias y el baseline entre 8:")
    w("  compare el equity TOTAL, no por estrategia.")
    w("- Profit factor y win rate con pocos trades no son estadisticamente fiables.")
    w("- Indicadores calculados una vez sobre toda la historia (en vivo: sobre la")
    w("  ventana de 600 velas); diferencias minimas en EMAs largas.")
    return "\n".join(out)


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=500, help="days to replay (365-500 recommended)")
    ap.add_argument("--out", default="reports", help="output folder")
    ap.add_argument("--no-learning", action="store_true", help="disable tuner + evaluator (sanity check)")
    ap.add_argument("--funds", type=float, default=None, help="paper funds (default INITIAL_CAPITAL)")
    ap.add_argument("--budget", type=float, default=None, help="risk budget (default 10%% of funds)")
    ap.add_argument("--aggressiveness", type=int, default=None, help="1..10 (default RISK_AGGRESSIVENESS)")
    ap.add_argument("--eval-interval-hours", type=float, default=168,
                    help="strategy evaluator cadence in simulated hours (live: 24; default 168 "
                         "keeps a 500-day replay around an hour)")
    args = ap.parse_args()

    import main   # noqa: F401 — its import installs console/file log handlers; override below

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    log_path = os.path.join(args.out, f"replay_{stamp}.log")
    root = logging.getLogger()
    for h in root.handlers:
        h.close()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s — %(message)s"))
    root.handlers = [file_handler]
    root.setLevel(logging.INFO)
    config.ANTHROPIC_API_KEY = ""   # the replay never calls any external API

    print(f"Descargando velas reales de Binance ({args.days} dias + historia)...", flush=True)
    data = fetch_data(args.days, os.path.join(args.out, "cache"))
    end = data["1h"].index[-1] + DELTA["1h"]
    start = end - pd.Timedelta(days=args.days)
    print(f"Replay {start:%Y-%m-%d} -> {end:%Y-%m-%d} ({len(data['1h'])} velas 1h, "
          f"{len(data['4h'])} 4h, {len(data['1d'])} 1d)", flush=True)

    def progress(ts, i, n, secs):
        print(f"  {ts:%Y-%m-%d}  {i / max(n, 1):5.1%}  ({secs / 60:.1f} min)", flush=True)

    db_path = os.path.join(args.out, f"replay_{stamp}.db")
    res = run_replay(data, start, end, db_path, learning=not args.no_learning, progress=progress,
                     eval_interval_hours=args.eval_interval_hours, funds=args.funds,
                     budget=args.budget, aggressiveness=args.aggressiveness)
    from binance_client import BINANCE_PUBLIC_BASE
    report = build_report(res, BINANCE_PUBLIC_BASE)
    report_path = os.path.join(args.out, f"REPLAY_REPORT_{stamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    pd.DataFrame(res["equity"], index=res["index"]).to_csv(os.path.join(args.out, f"replay_equity_{stamp}.csv"))
    print("\n" + report)
    print(f"\nReporte: {report_path}\nBase de datos: {db_path}\nLog: {log_path}")


if __name__ == "__main__":
    main_cli()
