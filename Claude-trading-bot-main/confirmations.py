"""
Confirmation engine — more reasons to agree before the learning book trades
────────────────────────────────────────────────────────────────────────────
After the strategy evaluator allows a signal, the learning book also asks
independent checks. Each votes +1 (supports the trade), 0 (neutral or no
data) or -1 (against); some -1 are a VETO:

  technical (analitico)
    tendencia diaria     1d close vs a rising / falling EMA-50
    tendencia 4h         same on 4h candles
    volatilidad          1d ATR in the top 10% of the last 180 days: against
    amplitud             share of the traded coins above their 1d EMA-50
  sentiment
    euforia / panico     Fear & Greed >= 80 or funding >= 0.05% / 8 h against
                         longs (<= 20, <= -0.03% against shorts)
    top traders          Binance + OKX top traders and Hyperliquid whales (majority)
  fundamental
    tono de noticias     mean headline tone of the last 24 h (>= 3 headlines)
    alertas de noticias  a SEVERE headline (hack, lawsuit, delisting...) on the
                         coin in the last 24 h: VETO for longs
    calendario macro     high-impact US event from 2 h before to 1 h after: VETO
    liquidez stablecoins 30-day change of the stablecoin supply (+1.5% / -1%)
  claude
    revision de Claude   the latest scheduled Claude view: 'a_favor' +1,
                         'bloquear' VETO; expired views are ignored

The trade goes ahead when nothing vetoes it and (votes for - votes against)
reaches the minimum of the aggressiveness (1-3: 2, 4-7: 1, 8-10: 0). The
checklist is kept with the position, so later reports can compare trades by
their confirmations; the lab book trades without this filter, which shows
what the blocked signals would have done.

Headlines, the macro calendar and Claude's view only exist from the first run
on: the replay can only test the technical, sentiment and stablecoin checks.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from strategies.base_strategy import SignalType
from utils import coin_of, symbol_of


@dataclass
class Check:
    name: str
    kind: str          # analitico | sentimiento | fundamental | claude
    vote: int          # +1 for, 0 neutral / no data, -1 against
    detail: str
    veto: bool = False


def _ok(df, n: int) -> bool:
    return df is not None and len(df) >= n


def _trend(df, span: int = 50) -> int:
    """+1 above a rising EMA, -1 below a falling one, 0 otherwise / no data."""
    if not _ok(df, span + 6):
        return 0
    ema = df["close"].ewm(span=span, adjust=False).mean()
    c = float(df["close"].iloc[-1])
    if c > ema.iloc[-1] and ema.iloc[-1] > ema.iloc[-6]:
        return 1
    if c < ema.iloc[-1] and ema.iloc[-1] < ema.iloc[-6]:
        return -1
    return 0


def _last(df, col):
    if df is None or col not in df or df[col].isna().all():
        return None
    v = df[col].dropna()
    return float(v.iloc[-1]) if len(v) else None


class ConfirmationEngine:

    def __init__(self, dfs_fn: Callable[[str, str], Optional[pd.DataFrame]], symbols: List[str],
                 clock: Callable[[], datetime] = None):
        """dfs_fn(symbol, interval) -> closed candles with market-data columns (or None)."""
        self.dfs_fn = dfs_fn
        self.symbols = list(symbols)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _df(self, symbol, interval):
        try:
            return self.dfs_fn(symbol, interval)
        except Exception:
            return None

    # ─── Checks ──────────────────────────────────────────────────────────────

    def checks(self, strategy, signal) -> List[Check]:
        sym = symbol_of(strategy)
        side = 1 if signal.type == SignalType.BUY else -1
        d1, h4, h1 = self._df(sym, "1d"), self._df(sym, "4h"), self._df(sym, "1h")
        now = self.clock()
        out = []

        t = _trend(d1)
        out.append(Check("tendencia diaria", "analitico", t * side,
                         {1: "precio sobre la EMA-50 diaria que sube", -1: "precio bajo la EMA-50 diaria que baja",
                          0: "sin tendencia diaria clara"}[t]))
        t = _trend(h4)
        out.append(Check("tendencia 4h", "analitico", t * side,
                         {1: "4h sobre su EMA-50 que sube", -1: "4h bajo su EMA-50 que baja",
                          0: "sin tendencia clara en 4h"}[t]))
        out.append(self._volatility(d1))
        out.append(self._breadth(side))
        out.append(self._crowd(h1 if h1 is not None else d1, side))
        out.append(self._top_traders(h1, side))
        out.append(self._news_tone(h1, side))
        out.append(self._alerts(coin_of(sym), side, now))
        out.append(self._macro(now))
        out.append(self._stablecoins(d1, side))
        out.append(self._claude(coin_of(sym), side, now))
        return out

    def _volatility(self, d1) -> Check:
        if not _ok(d1, 200):
            return Check("volatilidad", "analitico", 0, "sin datos suficientes")
        h, l, c = d1["high"], d1["low"], d1["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = (tr.ewm(alpha=1 / 14, adjust=False).mean() / c).iloc[-180:]
        pct = float((atr <= atr.iloc[-1]).mean())
        if pct >= 0.9:
            return Check("volatilidad", "analitico", -1, f"volatilidad extrema (percentil {pct:.0%} de 180 dias)")
        return Check("volatilidad", "analitico", 0, f"volatilidad normal (percentil {pct:.0%})")

    def _breadth(self, side) -> Check:
        states = []
        for s in self.symbols:
            d = self._df(s, "1d")
            if _ok(d, 60):
                ema = d["close"].ewm(span=50, adjust=False).mean()
                states.append(float(d["close"].iloc[-1]) > float(ema.iloc[-1]))
        if len(states) < 3:
            return Check("amplitud", "analitico", 0, "pocas criptos con datos")
        frac = sum(states) / len(states)
        vote = 1 if frac >= 0.6 else -1 if frac <= 0.4 else 0
        return Check("amplitud", "analitico", vote * side,
                     f"{sum(states)} de {len(states)} criptos sobre su EMA-50 diaria")

    def _crowd(self, df, side) -> Check:
        fng, fund = _last(df, "fng"), _last(df, "funding_rate")
        if fng is None and fund is None:
            return Check("euforia / panico", "sentimiento", 0, "sin datos")
        against = []
        if side > 0:
            if fng is not None and fng >= 80:
                against.append(f"codicia extrema ({fng:.0f})")
            if fund is not None and fund >= 0.0005:
                against.append(f"funding caro {fund * 100:+.3f}%")
        else:
            if fng is not None and fng <= 20:
                against.append(f"miedo extremo ({fng:.0f})")
            if fund is not None and fund <= -0.0003:
                against.append(f"funding negativo {fund * 100:+.3f}%")
        if against:
            return Check("euforia / panico", "sentimiento", -1, ", ".join(against))
        return Check("euforia / panico", "sentimiento", 0,
                     f"sin extremos (F&G {fng:.0f})" if fng is not None else "sin extremos")

    def _top_traders(self, h1, side) -> Check:
        from strategies.smart_money_catalog import SmartMoneyConsensusStrategy
        if h1 is None:
            return Check("top traders", "sentimiento", 0, "sin datos")
        votes = SmartMoneyConsensusStrategy().votes(h1)
        if not votes:
            return Check("top traders", "sentimiento", 0, "sin datos de posicionamiento")
        total = sum(votes.values())
        vote = (1 if total > 0 else -1 if total < 0 else 0) * side
        text = ", ".join(f"{k} {'+' if v > 0 else '-' if v < 0 else '='}" for k, v in votes.items())
        return Check("top traders", "sentimiento", vote, text)

    def _news_tone(self, h1, side) -> Check:
        tone, count = _last(h1, "news_tone"), _last(h1, "news_count")
        if tone is None or not count or count < 3:
            return Check("tono de noticias", "fundamental", 0, "pocas noticias en 24 h")
        vote = 1 if tone >= 0.25 else -1 if tone <= -0.25 else 0
        return Check("tono de noticias", "fundamental", vote * side,
                     f"tono {tone:+.2f} en {count:.0f} titulares de 24 h")

    def _alerts(self, coin, side, now) -> Check:
        from news_data import severe_alerts
        try:
            alerts = severe_alerts(coin, now - timedelta(hours=24))
        except Exception:
            alerts = []
        if alerts and side > 0:
            return Check("alertas de noticias", "fundamental", -1,
                         f"noticia grave: {alerts[0]['title'][:90]}", veto=True)
        return Check("alertas de noticias", "fundamental", 0,
                     f"{len(alerts)} alerta(s) graves" if alerts else "sin alertas graves")

    def _macro(self, now) -> Check:
        from news_data import macro_events_near
        try:
            events = macro_events_near(now)
        except Exception:
            events = []
        if events:
            e = events[0]
            return Check("calendario macro", "fundamental", -1,
                         f"evento de alto impacto: {e['title']} ({e['ts'][11:16]} UTC)", veto=True)
        return Check("calendario macro", "fundamental", 0, "sin eventos de alto impacto cerca")

    def _stablecoins(self, d1, side) -> Check:
        if d1 is None or "stable_supply" not in d1:
            return Check("liquidez stablecoins", "fundamental", 0, "sin datos")
        s = d1["stable_supply"].dropna()
        if len(s) < 31:
            return Check("liquidez stablecoins", "fundamental", 0, "sin datos suficientes")
        chg = float(s.iloc[-1]) / float(s.iloc[-31]) - 1
        vote = 1 if chg >= 0.015 else -1 if chg <= -0.01 else 0
        return Check("liquidez stablecoins", "fundamental", vote * side,
                     f"stablecoins {chg:+.1%} en 30 dias")

    def _claude(self, coin, side, now) -> Check:
        from claude_view import get_view
        v = get_view(coin, now)
        if v is None:
            return Check("revision de Claude", "claude", 0, "sin revision vigente")
        stance = v["long"] if side > 0 else v["short"]
        nota = v.get("nota") or ""
        if stance == "bloquear":
            return Check("revision de Claude", "claude", -1, f"Claude bloquea: {nota}", veto=True)
        if stance == "a_favor":
            return Check("revision de Claude", "claude", 1, f"Claude a favor: {nota}")
        return Check("revision de Claude", "claude", 0, f"Claude neutral: {nota}")

    # ─── Decision ────────────────────────────────────────────────────────────

    @staticmethod
    def decide(checks: List[Check], min_net: int):
        """(ok, reason, summary) — reason explains a block in Spanish."""
        pro = [c for c in checks if c.vote > 0]
        con = [c for c in checks if c.vote < 0]
        net = len(pro) - len(con)
        summary = {"a_favor": [c.name for c in pro], "en_contra": [c.name for c in con],
                   "neto": net, "minimo": min_net,
                   "detalle": {c.name: f"{c.vote:+d} {c.detail}" for c in checks}}
        vetoes = [c for c in con if c.veto]
        if vetoes:
            return False, "confirmaciones: VETO - " + "; ".join(
                f"{c.name}: {c.detail}" for c in vetoes), summary
        if net < min_net:
            against = "; ".join(f"{c.name} ({c.detail})" for c in con) or "ninguna en contra"
            return False, (f"confirmaciones: faltan confirmaciones ({len(pro)} a favor, {len(con)} en contra, "
                           f"neto {net} < {min_net}) - en contra: {against}"), summary
        return True, "", summary
