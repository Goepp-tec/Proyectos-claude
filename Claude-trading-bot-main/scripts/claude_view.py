"""
Claude's scheduled review — context in, view out (runs inside the container)
────────────────────────────────────────────────────────────────────────────
  python scripts/claude_view.py context
      What the learning book did lately (signals and why they were taken or
      blocked, open positions, closed trades), the evaluator's ratings, the
      latest headlines and severe alerts per coin, upcoming US macro events,
      the market snapshot and the current views. Read-only.

  python scripts/claude_view.py set --coin BTC --long bloquear --short neutral \\
         --hours 8 --nota "motivo breve"
      Leaves Claude's view for a coin: 'a_favor' / 'neutral' / 'bloquear' per
      side. It only adds a vote or a veto to the confirmation engine; it never
      opens or closes a position and expires after --hours (max 48).

  python scripts/claude_view.py show
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import database as db
from claude_view import VALUES, all_views, set_view
from utils import coin_of


def _learn_book() -> str:
    return "observe" if db.get_meta("trading_mode") == "OBSERVE" else "main"


def build_context(now: datetime = None, hours: int = 12) -> str:
    from market_data import load_series
    from news_data import macro_events_near, recent_headlines
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)
    book = _learn_book()
    out = []
    w = out.append
    w(f"CONTEXTO PARA LA REVISION DE CLAUDE — {now:%Y-%m-%d %H:%M} UTC (paper trading)")
    w(f"Libro que aprende: '{book}' | criptos: {', '.join(coin_of(s) for s in config.SYMBOLS)}")
    try:
        from risk_engine import RiskSettings, profile
        s = RiskSettings.load()
        p = profile(s.aggressiveness)
        w(f"Presupuesto ${s.budget:,.0f}, agresividad {p['level']}/10 (evaluador: {', '.join(p['statuses'])}; "
          f"confirmaciones netas minimas: {p['min_confirmations']}), modo {s.mode}")
    except Exception:
        pass

    w("")
    w(f"SENALES DEL LIBRO QUE APRENDE (ultimas {hours} h)")
    rows = db.get_conn().execute(
        "SELECT * FROM signal_log WHERE book=? AND recorded_at >= ? ORDER BY recorded_at DESC LIMIT 40",
        (book, since.isoformat())).fetchall()
    for r in rows:
        state = "ABIERTA" if r["acted"] else "bloqueada"
        w(f"  {r['recorded_at'][5:16]} {r['strategy_name']:<28} {r['signal_type']:<4} {state:<9} "
          f"{(r['reason'] or '')[:110]}")
    if not rows:
        w("  ninguna")

    w("POSICIONES ABIERTAS")
    positions = db.get_open_positions(book=book)
    for p in positions:
        meta = p.get("metadata") or {}
        conf = (meta.get("confirmaciones") or {}) if isinstance(meta, dict) else {}
        w(f"  {p['strategy_name']:<28} {p['side']:<5} entrada {p['entry_price']:,.4f} SL {p['stop_loss']:,.4f} "
          f"TP {p['take_profit']:,.4f} desde {str(p.get('entry_time', ''))[5:16]}"
          + (f" | confirmaciones neto {conf.get('neto')}" if conf else ""))
    if not positions:
        w("  ninguna")

    w("TRADES CERRADOS (ultimos 3 dias)")
    t0 = (now - timedelta(days=3)).isoformat()
    trades = [t for t in db.get_trades(limit=10**6, book=book) if (t["closed_at"] or "") >= t0]
    for t in trades[:20]:
        w(f"  {t['closed_at'][5:16]} {t['strategy_name']:<28} {t['side']:<5} P&L ${t['pnl']:+.2f} "
          f"({t['pnl_pct']:+.2%}) {t['exit_reason']}")
    if trades:
        w(f"  total: {len(trades)} trades, {sum(t['pnl'] > 0 for t in trades)} con ganancia, "
          f"P&L ${sum(t['pnl'] for t in trades):+.2f}")
    else:
        w("  ninguno")

    w("EVALUADOR: estrategias que hoy pueden operar (VIABLE / CONDICIONAL)")
    ok = [s for s in db.get_all_strategy_status() if s["status"] in ("VIABLE", "CONDICIONAL")]
    for s in ok[:25]:
        w(f"  {s['strategy_name']:<28} {s['status']:<12} {(s['reason'] or '')[:70]}")
    if not ok:
        w("  ninguna")

    w("")
    w("NOTICIAS (ultimas 24 h) Y MERCADO POR CRIPTO")
    news = recent_headlines(now - timedelta(hours=24))
    last = lambda sym, m: (lambda s: float(s.iloc[-1]) if len(s) else None)(load_series(sym, m))
    for sym in config.SYMBOLS:
        coin = coin_of(sym)
        mine = [n for n in news if coin in n["coins"].split(",")]
        severe = [n for n in mine if n["severe"]]
        parts = []
        for label, metric, fmt in (("F&G", "fng", "{:.0f}"), ("funding", "funding_rate", "{:+.4%}"),
                                   ("ballenas HL", "hl_top_net", "{:+.0%}"), ("tono", "news_tone", "{:+.2f}")):
            v = last(sym, metric)
            if v is not None:
                parts.append(f"{label} {fmt.format(v)}")
        w(f"  {coin}: {len(mine)} titulares, {len(severe)} graves | " + " | ".join(parts))
        for n in (severe + [n for n in mine if not n["severe"]])[:4]:
            w(f"     {'GRAVE ' if n['severe'] else ''}{n['ts'][5:16]} [{n['source']}] {n['title'][:100]}")

    w("CALENDARIO MACRO EE.UU. (alto impacto, proximas 48 h)")
    events = macro_events_near(now, before_min=48 * 60, after_min=0)
    for e in events:
        w(f"  {e['ts'][5:16]} UTC  {e['title']}")
    if not events:
        w("  ninguno (o calendario sin datos)")

    w("VISTAS DE CLAUDE ACTUALES")
    views = all_views(now)
    for coin, v in views.items():
        w(f"  {coin}: largos {v['long']}, cortos {v['short']} ({'vigente' if v['vigente'] else 'vencida'} "
          f"hasta {v['expires_at'][5:16]}) {v.get('nota', '')}")
    if not views:
        w("  ninguna")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("context")
    c.add_argument("--hours", type=int, default=12)
    s = sub.add_parser("set")
    s.add_argument("--coin", required=True)
    s.add_argument("--long", choices=VALUES, default="neutral")
    s.add_argument("--short", choices=VALUES, default="neutral")
    s.add_argument("--hours", type=float, default=8)
    s.add_argument("--nota", default="")
    sub.add_parser("show")
    args = ap.parse_args(argv)
    db.init_db()
    if args.cmd == "context":
        print(build_context(hours=args.hours))
    elif args.cmd == "set":
        v = set_view(args.coin, long=args.long, short=args.short, nota=args.nota, hours=args.hours)
        print(f"{args.coin.upper()}: largos {v['long']}, cortos {v['short']}, hasta {v['expires_at'][:16]} UTC")
    else:
        views = all_views()
        for coin, v in views.items():
            print(f"{coin}: largos {v['long']}, cortos {v['short']} — {'vigente' if v['vigente'] else 'vencida'} "
                  f"hasta {v['expires_at'][:16]} UTC — {v.get('nota', '')}")
        if not views:
            print("sin vistas de Claude")


if __name__ == "__main__":
    main()
