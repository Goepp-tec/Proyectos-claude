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
TUNED_BOOK = "tuned"     # replay only: tuned params + fixed ML confidence

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
    history_days = replay_days + config.LEARNING_PROPOSAL_DAYS + \
        config.LEARNING_VALIDATION_DAYS + config.LEARNING_WARMUP_DAYS + 10
    os.makedirs(cache_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(cache_dir, f"klines_{config.SYMBOL}_{replay_days}d_{stamp}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    fetcher = BinancePublicDataFetcher(BINANCE_PUBLIC_BASE)
    data = {
        "1h": fetcher.get_klines_since(config.SYMBOL, "1h", replay_days + 2),
        "4h": fetcher.get_klines_since(config.SYMBOL, "4h", history_days),
        "1d": fetcher.get_klines_since(config.SYMBOL, "1d", history_days),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


# ─── Replay core ──────────────────────────────────────────────────────────────

def run_replay(data: dict, start: pd.Timestamp, end: pd.Timestamp, db_path: str,
               learning: bool = True, progress=None) -> dict:
    import main
    from adaptive_tuner import AdaptiveTuner
    from learning_engine import LearningEngine
    from strategies import ALL_STRATEGIES

    config.DB_PATH = db_path
    conn = getattr(db._local, "conn", None)
    if conn is not None:
        conn.close()
    db._local.conn = None
    db.init_db()

    clock = {"now": start}
    iso_clock = lambda: clock["now"].isoformat()

    learners = [S() for S in ALL_STRATEGIES]
    for s in learners:
        s.is_active = True
    baselines = main.build_baselines({s.name for s in learners})

    # The live TradingBot without its network client: same methods, historical clock.
    bot = main.TradingBot.__new__(main.TradingBot)
    bot.strategies, bot.baselines = learners, baselines
    bot.book, bot.mode = LEARN_BOOK, "OBSERVE"
    bot._strat_dfs, bot._logged_candle, bot._current_price = {}, {}, 0.0
    bot.portfolio = main.make_portfolio(None, learners, LEARN_BOOK, clock=iso_clock)
    bot.baseline_portfolio = main.make_portfolio(None, baselines, "baseline", clock=iso_clock)
    bot.learning = LearningEngine({s.name: s for s in learners})
    # Replay-only 'tuned' book: the learners' (tuned) params with the baseline's
    # fixed ML confidence — isolates the effect of the parameter changes.
    tuned_pm = main.make_portfolio(None, learners, TUNED_BOOK, clock=iso_clock)

    def history(interval, days, now):
        df = data[interval]
        closed = df.iloc[:(df.index + DELTA[interval]).searchsorted(now, side="right")]
        return closed[closed.index >= now - pd.Timedelta(days=days)]

    bot.tuner = AdaptiveTuner(
        learners={s.name: s for s in learners}, history_fn=history,
        equity_fn=bot._strategy_equity, clock=lambda: clock["now"], learner_book=LEARN_BOOK,
    )

    close_times = {iv: data[iv].index + DELTA[iv] for iv in ("4h", "1d")}
    last_closed = {iv: -1 for iv in close_times}
    hours = data["1h"][(data["1h"].index >= start) & (data["1h"].index < end)]
    equity = {"learn": [], "tuned": [], "baseline": []}
    books = (("learn", bot.portfolio, learners), ("tuned", tuned_pm, learners),
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
        changed = set()
        for iv, closes in close_times.items():
            n = closes.searchsorted(now, side="right")
            if n != last_closed[iv]:
                last_closed[iv] = n
                bot._strat_dfs[iv] = data[iv].iloc[max(0, n - config.LOOKBACK_CANDLES):n]
                changed.add(iv)
        if changed:
            bot._process_signals([s for s in learners if s.candle_interval in changed],
                                 bot.portfolio, price,
                                 lambda name, df: bot.learning.get_confidence(name, df))
            bot._process_signals([s for s in learners if s.candle_interval in changed],
                                 tuned_pm, price,
                                 lambda name, df: config.BASELINE_ML_CONFIDENCE)
            bot._process_signals([b for b in baselines if b.candle_interval in changed],
                                 bot.baseline_portfolio, price,
                                 lambda name, df: config.BASELINE_ML_CONFIDENCE)

        # 3. Journal + learning, exactly as the live learning loop
        bot._journal_new_trades()
        if learning:
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
        "learn": _metrics(pd.Series(equity["learn"], index=idx), LEARN_BOOK),
        "tuned": _metrics(pd.Series(equity["tuned"], index=idx), TUNED_BOOK),
        "baseline": _metrics(pd.Series(equity["baseline"], index=idx), "baseline"),
    }
    return {
        "metrics": metrics, "equity": equity, "index": idx, "per_strategy": per_strategy,
        "learners": learners, "baselines": baselines,
        "btc_return": final_price / float(hours["open"].iloc[0]) - 1,
        "start": start, "end": end, "elapsed_sec": time.time() - t0,
    }


def _metrics(eq: pd.Series, book: str) -> dict:
    start_eq = config.INITIAL_CAPITAL
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
    L, T, B = res["metrics"]["learn"], res["metrics"]["tuned"], res["metrics"]["baseline"]
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
    w(f"Capital inicial : ${config.INITIAL_CAPITAL:,.0f} por libro, repartido en 8 estrategias")
    w(f"BTC buy & hold  : {res['btc_return']:+.2%} en el mismo periodo (referencia)")
    w(f"Duracion calculo: {res['elapsed_sec'] / 60:.1f} min")
    w("")
    w("APRENDE      = parametros del tuner + confianza ML adaptativa (lo que corre en vivo)")
    w("SOLO AJUSTES = parametros del tuner + confianza fija 0.55 (aisla el efecto de los ajustes)")
    w("BASELINE     = parametros por defecto congelados + confianza fija 0.55")
    w("")
    w(f"{'Metrica':<34}{'APRENDE':>15}{'SOLO AJUSTES':>15}{'BASELINE':>15}")
    w("-" * 79)
    for label, fmt, key in rows:
        w(f"{label:<34}{fmt.format(L[key]):>15}{fmt.format(T[key]):>15}{fmt.format(B[key]):>15}")
    w("")
    for name, M in (("Aprende", L), ("Solo ajustes", T)):
        diff = M["equity_end"] - B["equity_end"]
        better = diff > 0 and M["max_drawdown"] <= B["max_drawdown"] + 0.02
        verdict = "SUPERO" if better else "NO supero"
        w(f"{name:<13} vs baseline: equity {diff:+,.2f} USD -> {verdict} al baseline.")
        if B["trades"] and M["trades"] < 0.5 * B["trades"]:
            w(f"  OJO: hizo {M['trades']} trades vs {B['trades']} del baseline; la diferencia se")
            w("  explica sobre todo por operar menos, no por operar mejor.")
    w("")
    w("EQUITY FINAL POR ESTRATEGIA")
    w(f"{'Estrategia':<24}{'Aprende':>12}{'Solo ajust.':>12}{'Baseline':>12}")
    for name, eq_l in res["per_strategy"]["learn"].items():
        eq_t = res["per_strategy"]["tuned"][name]
        eq_b = res["per_strategy"]["baseline"][name]
        w(f"{name:<24}{eq_l:>12,.2f}{eq_t:>12,.2f}{eq_b:>12,.2f}")
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
    w("- Aprende vs Solo ajustes = efecto de la confianza ML (win rate reciente y rachas:")
    w("  ajusta el tamano y pausa entradas). Solo ajustes vs Baseline = efecto de los")
    w("  cambios de parametros. 'Solo ajustes' comparte instancias de estrategia con")
    w("  'Aprende' (mismos parametros en cada momento); existe solo en el replay.")
    w("- Profit factor y win rate con pocos trades no son estadisticamente fiables.")
    w("- Indicadores calculados una vez sobre toda la historia (en vivo: sobre la")
    w("  ventana de 600 velas); diferencias minimas en EMAs largas.")
    return "\n".join(out)


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=500, help="days to replay (365-500 recommended)")
    ap.add_argument("--out", default="reports", help="output folder")
    ap.add_argument("--no-learning", action="store_true", help="disable the tuner (sanity check)")
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
    res = run_replay(data, start, end, db_path, learning=not args.no_learning, progress=progress)
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
