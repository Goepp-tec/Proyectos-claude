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

Several symbols (--symbols BTCUSDT,ETHUSDT,...): every strategy runs on
every symbol, each on its own candles and prices, all on the same hourly
clock (the primary symbol's). The learning book shares one risk budget; lab
and baseline get INITIAL_CAPITAL per symbol (see main.book_capital).

Lab and baseline use simulated fills (price ± SLIPPAGE, fee 0.1%) — like the
live 'observe' and 'baseline' books. This is a replay/backtest, NOT live evidence.

Usage:
    python run_replay_test.py              # 500 days, SYMBOLS from .env
    python run_replay_test.py --days 365 --symbols BTCUSDT,ETHUSDT
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

def fetch_data(replay_days: int, cache_dir: str, symbols=None) -> dict:
    """Real Binance candles {symbol: {interval: df}} (closed only), cached per day."""
    from binance_client import BINANCE_PUBLIC_BASE, BinancePublicDataFetcher
    history_days = replay_days + max(
        config.LEARNING_PROPOSAL_DAYS + config.LEARNING_VALIDATION_DAYS + config.LEARNING_WARMUP_DAYS,
        config.EVAL_WINDOWS * config.EVAL_WINDOW_DAYS + config.EVAL_WARMUP_DAYS) + 10
    os.makedirs(cache_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    fetcher = BinancePublicDataFetcher(BINANCE_PUBLIC_BASE)
    out = {}
    for symbol in symbols or config.SYMBOLS:
        path = os.path.join(cache_dir, f"klines_{symbol}_{replay_days}d_{history_days}h_{stamp}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                out[symbol] = pickle.load(f)
            continue
        # 1h is both the replay clock and the interval of some catalog strategies
        out[symbol] = {iv: fetcher.get_klines_since(symbol, iv, history_days) for iv in ("1h", "4h", "1d")}
        with open(path, "wb") as f:
            pickle.dump(out[symbol], f)
    return out


# ─── Replay core ──────────────────────────────────────────────────────────────

def run_replay(data: dict, start: pd.Timestamp, end: pd.Timestamp, db_path: str,
               learning: bool = True, progress=None, eval_interval_hours: float = None,
               funds: float = None, budget: float = None, aggressiveness: int = None,
               collect_market: bool = True, confirmations: bool = True) -> dict:
    import main
    from adaptive_tuner import AdaptiveTuner
    from confirmations import ConfirmationEngine
    from learning_engine import LearningEngine
    from market_data import MarketDataCollector, enrich
    from risk_engine import RiskEngine, RiskSettings
    from strategies import ALL_STRATEGIES, CANDIDATE_STRATEGIES, build_line_up
    from strategy_evaluator import StrategyEvaluator
    from utils import symbol_of

    # {interval: df} (one symbol) or {symbol: {interval: df}}; primary symbol first
    if "1h" in data:
        data = {config.SYMBOL: data}
    symbols = sorted(data, key=lambda sym: (sym != config.SYMBOL, list(data).index(sym)))
    primary = symbols[0]

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

    # Free market data (funding and Fear & Greed have years of history; the
    # futures ratios only the last 30 days), joined to every candle as of its
    # close time — strategies never see a value published after the candle.
    if collect_market:
        MarketDataCollector(symbols).update(force=True)
    data = {sym: {iv: enrich(df, sym, iv) for iv, df in ivs.items()} for sym, ivs in data.items()}

    clock = {"now": start}
    iso_clock = lambda: clock["now"].isoformat()

    # Same line-up as the live bot: registered strategies + candidate catalog, per symbol.
    learners = build_line_up(ALL_STRATEGIES + CANDIDATE_STRATEGIES, symbols)
    for s in learners:
        s.is_active = True
    registered = {S().name for S in ALL_STRATEGIES}
    baselines = main.build_baselines({s.name for s in learners if s.base_name in registered}, symbols)
    lab = main.build_lab(learners)
    book_capital = main.book_capital(symbols)

    # The live TradingBot without its network client: same methods, historical clock.
    bot = main.TradingBot.__new__(main.TradingBot)
    bot.strategies, bot.baselines, bot.lab = learners, baselines, lab
    bot.symbols = symbols
    bot.book, bot.mode = LEARN_BOOK, "OBSERVE"
    bot._strat_dfs, bot._logged_candle, bot._current_price, bot._prices = {}, {}, 0.0, {}
    bot.risk = RiskEngine(LEARN_BOOK, clock=lambda: clock["now"])
    bot.portfolio = main.make_portfolio(None, learners, LEARN_BOOK, clock=iso_clock,
                                        capital_base=settings.budget, risk_engine=bot.risk)
    bot.baseline_portfolio = main.make_portfolio(None, baselines, "baseline", clock=iso_clock,
                                                 capital_base=book_capital)
    bot.lab_portfolio = main.make_portfolio(None, lab, "lab", clock=iso_clock, capital_base=book_capital)
    bot.learning = LearningEngine({s.name: s for s in learners})

    def history(interval, days, now, symbol=None):
        df = data[symbol or primary][interval]
        closed = df.iloc[:(df.index + DELTA[interval]).searchsorted(now, side="right")]
        return closed[closed.index >= now - pd.Timedelta(days=days)]

    bot.tuner = AdaptiveTuner(
        learners={s.name: s for s in learners if s.base_name in registered}, history_fn=history,
        equity_fn=bot._strategy_equity, clock=lambda: clock["now"], learner_book=LEARN_BOOK,
    )
    bot.evaluator = StrategyEvaluator({s.name: s for s in learners}, history_fn=history,
                                      clock=lambda: clock["now"], live_book="lab")
    # Confirmations on replay time: headlines, the macro calendar and Claude's
    # view have no history, so only technical / sentiment / stablecoin checks vote.
    bot.confirm = (ConfirmationEngine(dfs_fn=lambda sym, iv: bot._strat_dfs.get((sym, iv)),
                                      symbols=symbols, clock=lambda: clock["now"])
                   if confirmations else None)

    pairs = sorted({(symbol_of(s), s.candle_interval) for s in learners})
    close_times = {(sym, iv): data[sym][iv].index + DELTA[iv] for sym, iv in pairs}
    last_closed = {k: -1 for k in close_times}
    hours = data[primary]["1h"][(data[primary]["1h"].index >= start) & (data[primary]["1h"].index < end)]
    # every symbol on the primary's hourly clock (a missing hour = no price that hour)
    bars = {sym: data[sym]["1h"].reindex(hours.index)[["open", "high", "low", "close"]].to_numpy()
            for sym in symbols}
    first_open = {sym: next((float(r[0]) for r in bars[sym] if not np.isnan(r[0])), float("nan"))
                  for sym in symbols}
    last_price = {}
    equity = {"learn": [], "lab": [], "baseline": []}
    books = (("learn", bot.portfolio, learners), ("lab", bot.lab_portfolio, lab),
             ("baseline", bot.baseline_portfolio, baselines))
    t0 = time.time()

    for i, ts in enumerate(hours.index):
        # 1. SL/TP inside the hour, each symbol along its own intra-hour path
        clock["now"] = ts + pd.Timedelta(minutes=30)
        open_positions = db.get_open_positions(book=None)
        for sym in symbols:
            o, h, l, c = bars[sym][i]
            if np.isnan(c):
                continue
            levels = [lvl for p in open_positions if p["symbol"] == sym
                      for lvl in (p["stop_loss"], p["take_profit"]) if lvl]
            if not levels:
                continue
            for price in intrabar_path(o, h, l, c, levels):
                if sym == primary:
                    bot._current_price = price
                for _, pm, _ in books:
                    pm.check_open_positions({sym: price})

        # 2. Hour closed: new 4h / 1d candles → signals at the close prices
        now = ts + DELTA["1h"]
        last_price.update({sym: float(bars[sym][i][3]) for sym in symbols if not np.isnan(bars[sym][i][3])})
        prices = dict(last_price)
        clock["now"], bot._current_price, bot._prices = now, prices.get(primary, 0.0), prices
        bot._risk_housekeeping(prices)         # day / kill-switch state, like the live loop
        changed = set()
        for key, closes in close_times.items():
            n = closes.searchsorted(now, side="right")
            if n != last_closed[key]:
                last_closed[key] = n
                sym, iv = key
                bot._strat_dfs[key] = data[sym][iv].iloc[max(0, n - config.LOOKBACK_CANDLES):n]
                changed.add(key)
        if changed:
            fixed = lambda name, df: config.BASELINE_ML_CONFIDENCE
            due = lambda strats: [s for s in strats if (symbol_of(s), s.candle_interval) in changed]
            bot._process_signals(due(learners), bot.portfolio, prices,
                                 lambda name, df: bot.learning.get_confidence(name, df),
                                 gate=bot._learner_gate)
            bot._process_signals(due(baselines), bot.baseline_portfolio, prices, fixed)
            bot._process_signals(due(lab), bot.lab_portfolio, prices, fixed, gate=bot._lab_gate)

        # 3. Journal + evaluator + learning, exactly as the live learning loop
        bot._journal_new_trades()
        if learning:
            bot.evaluator.run_cycle_if_due()
            bot.tuner.run_cycle_if_due()

        for label, pm, strats in books:
            equity[label].append(sum(pm.strategy_equity(s.name, prices) for s in strats))
        if progress and i % (24 * 30) == 0:
            progress(ts, i, len(hours), time.time() - t0)

    final_prices = dict(last_price)
    per_strategy = {
        label: {s.name: pm.strategy_equity(s.name, final_prices) for s in strats}
        for label, pm, strats in books
    }
    idx = hours.index + DELTA["1h"]
    metrics = {
        "learn": _metrics(pd.Series(equity["learn"], index=idx), LEARN_BOOK, settings.budget),
        "lab": _metrics(pd.Series(equity["lab"], index=idx), "lab", book_capital),
        "baseline": _metrics(pd.Series(equity["baseline"], index=idx), "baseline", book_capital),
    }
    buy_hold = {sym: final_prices[sym] / first_open[sym] - 1 for sym in symbols
                if sym in final_prices and first_open[sym] > 0}
    risk_events = {
        "kill_switch": db.get_meta(f"risk:{LEARN_BOOK}:kill"),
        "blocked": {r["reason"].split("(")[0].strip(): r["n"] for r in db.get_conn().execute(
            "SELECT reason, COUNT(*) AS n FROM signal_log WHERE book=? AND reason LIKE 'riesgo:%' "
            "GROUP BY reason", (LEARN_BOOK,)).fetchall()},
        "confirmations": confirmations,
        "blocked_by_confirmations": db.get_conn().execute(
            "SELECT COUNT(*) FROM signal_log WHERE book=? AND reason LIKE 'confirmaciones:%'",
            (LEARN_BOOK,)).fetchone()[0],
    }
    return {
        "metrics": metrics, "equity": equity, "index": idx, "per_strategy": per_strategy,
        "learners": learners, "baselines": baselines, "settings": settings, "risk_events": risk_events,
        "btc_return": buy_hold.get(primary, float("nan")),
        "buy_hold": buy_hold, "basket_return": float(np.mean(list(buy_hold.values()))),
        "symbols": symbols, "book_capital": book_capital,
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
    syms = res.get("symbols", [config.SYMBOL])
    w(f"Datos           : {data_source} (velas reales {', '.join(syms)} 1h/4h/1d)")
    s = res["settings"]
    from risk_engine import profile
    p = profile(s.aggressiveness)
    w(f"Fondos (paper)  : ${s.funds:,.0f}" + (f" x {len(syms)} criptos = ${res['book_capital']:,.0f}"
                                               if len(syms) > 1 else "") + "  (capital de LAB y BASELINE)")
    w(f"Presupuesto     : ${s.budget:,.0f}  (capital de APRENDE)  |  agresividad {p['level']}/10: "
      f"riesgo/operacion {p['risk_per_trade']:.2%}, max {p['max_open']} posiciones, "
      f"limite diario {p['daily_loss']:.1%}, freno {p['max_drawdown']:.0%}, "
      f"opera: {', '.join(p['statuses'])}")
    w(f"BTC buy & hold  : {res['btc_return']:+.2%} en el mismo periodo (referencia)")
    if len(syms) > 1:
        w(f"Canasta b&h     : {res['basket_return']:+.2%} (mismo peso en cada cripto: "
          + ", ".join(f"{sym[:-4]} {r:+.1%}" for sym, r in res["buy_hold"].items()) + ")")
    w(f"Evaluador       : cada {config.EVAL_INTERVAL_HOURS:g} h de tiempo simulado "
      f"(en vivo: cada 24 h); tuner cada {config.LEARNING_INTERVAL_HOURS:g} h")
    w(f"Duracion calculo: {res['elapsed_sec'] / 60:.1f} min")
    w("")
    n_all = len(res["learners"]) // max(len(syms), 1)
    w(f"APRENDE  = {n_all} estrategias x {len(syms)} cripto(s); solo opera lo que el evaluador")
    w("           aprueba para esta agresividad; tamano y limites del motor de riesgo sobre")
    w("           el PRESUPUESTO (uno solo para todas las criptos). Sus % son sobre el presupuesto.")
    w("LAB      = las mismas sin filtro (menos las DESCARTADAS): la evidencia en vivo")
    w("BASELINE = las 8 originales con parametros por defecto congelados, sin filtro")
    w("")
    w(f"{'Metrica':<34}{'APRENDE':>15}{'LAB':>15}{'BASELINE':>15}")
    w("-" * 79)
    for label, fmt, key in rows:
        w(f"{label:<34}{fmt.format(L[key]):>15}{fmt.format(T[key]):>15}{fmt.format(B[key]):>15}")
    w("")
    # Aprende trades the budget, baseline / lab the funds: compare returns, not dollars.
    for name, other in (("baseline", B), ("lab", T)):
        diff = L["total_return"] - other["total_return"]
        better = diff > 0 and L["max_drawdown"] <= other["max_drawdown"] + 0.02
        w(f"Aprende vs {name}: retorno {L['total_return']:+.2%} vs {other['total_return']:+.2%} "
          f"({diff * 100:+.2f} pts), max drawdown {L['max_drawdown']:.1%} vs {other['max_drawdown']:.1%} "
          f"-> {'SUPERO' if better else 'NO supero'}.")
    if B["trades"] and L["trades"] < 0.5 * B["trades"]:
        w(f"  OJO: hizo {L['trades']} trades vs {B['trades']} del baseline; buena parte de la")
        w("  diferencia viene de operar menos (el filtro evita estrategias no viables).")
    w("")
    if len(syms) > 1:
        w("RESULTADO POR CRIPTO (P&L de trades cerrados, USD)")
        w(f"{'Cripto':<10}{'Aprende':>12}{'Lab':>12}{'Baseline':>12}{'Trades aprende':>16}")
        by_sym = {}
        for book, label in ((LEARN_BOOK, "learn"), ("lab", "lab"), ("baseline", "baseline")):
            for t in db.get_trades(limit=10**7, book=book):
                d = by_sym.setdefault(t["symbol"], {"learn": 0.0, "lab": 0.0, "baseline": 0.0, "n": 0})
                d[label] += float(t["pnl"])
                d["n"] += label == "learn"
        for sym in syms:
            d = by_sym.get(sym, {"learn": 0.0, "lab": 0.0, "baseline": 0.0, "n": 0})
            w(f"{sym[:-4]:<10}{d['learn']:>12,.2f}{d['lab']:>12,.2f}{d['baseline']:>12,.2f}{d['n']:>16d}")
        w("")
    ev = res["risk_events"]
    w(f"CONFIRMACIONES (libro que aprende): {'activas' if ev.get('confirmations', True) else 'DESACTIVADAS'}; "
      f"senales bloqueadas por falta de confirmaciones: {ev.get('blocked_by_confirmations', 0)}")
    w("  (en el replay solo votan tecnico, sentimiento y stablecoins: noticias, calendario y")
    w("   Claude no tienen historia)")
    w("")
    w("MOTOR DE RIESGO (libro que aprende)")
    kill = json.loads(ev["kill_switch"] or "null")
    w(f"  Freno de emergencia: {'ACTIVADO el ' + kill['at'][:10] + ' - ' + kill['reason'] if kill else 'no se activo'}")
    if ev["blocked"]:
        w("  Entradas bloqueadas por riesgo:")
        for reason, n in sorted(ev["blocked"].items(), key=lambda x: -x[1])[:8]:
            w(f"    {n:>5} x {reason}")
    w("")
    w("EQUITY FINAL POR ESTRATEGIA (aprende / lab / baseline) Y CALIFICACION FINAL")
    w(f"{'Estrategia':<28}{'Aprende':>10}{'Lab':>10}{'Baseline':>10}  {'Estado':<12}{'Punt.':>6}  Motivo")
    status = {s["strategy_name"]: s for s in db.get_all_strategy_status()}
    for name, eq_l in res["per_strategy"]["learn"].items():
        eq_t = res["per_strategy"]["lab"].get(name, 0.0)
        eq_b = res["per_strategy"]["baseline"].get(name)
        st = status.get(name, {})
        w(f"{name:<28}{eq_l:>10,.2f}{eq_t:>10,.2f}{(f'{eq_b:,.2f}' if eq_b is not None else '-'):>10}  "
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
    w("- Aprende y lab reparten el capital entre todas las estrategias y el baseline")
    w("  entre las 8 originales: compare el equity TOTAL (en %), no por estrategia.")
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
    ap.add_argument("--no-confirmations", action="store_true",
                    help="learning book without the confirmation engine (to compare)")
    ap.add_argument("--symbols", default=None,
                    help="comma separated, e.g. BTCUSDT,ETHUSDT (default SYMBOLS from .env)")
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

    symbols = ([x.strip().upper() for x in args.symbols.split(",") if x.strip()]
               if args.symbols else config.SYMBOLS)
    print(f"Descargando velas reales de Binance ({', '.join(symbols)}; {args.days} dias + historia)...",
          flush=True)
    data = fetch_data(args.days, os.path.join(args.out, "cache"), symbols)
    first = data[symbols[0]]
    end = first["1h"].index[-1] + DELTA["1h"]
    start = end - pd.Timedelta(days=args.days)
    print(f"Replay {start:%Y-%m-%d} -> {end:%Y-%m-%d} ({len(first['1h'])} velas 1h, "
          f"{len(first['4h'])} 4h, {len(first['1d'])} 1d por cripto)", flush=True)

    def progress(ts, i, n, secs):
        print(f"  {ts:%Y-%m-%d}  {i / max(n, 1):5.1%}  ({secs / 60:.1f} min)", flush=True)

    db_path = os.path.join(args.out, f"replay_{stamp}.db")
    res = run_replay(data, start, end, db_path, learning=not args.no_learning, progress=progress,
                     eval_interval_hours=args.eval_interval_hours, funds=args.funds,
                     budget=args.budget, aggressiveness=args.aggressiveness,
                     confirmations=not args.no_confirmations)
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
