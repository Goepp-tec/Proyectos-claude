"""
Risk engine — aggressiveness 1-10, budget, daily loss limit, kill switch
────────────────────────────────────────────────────────────────────────
Applies to the book the bot actually trades (the learning book):

  • funds       : paper account size (display / sanity check)
  • budget      : the most the bot may use; the book's capital IS the budget
                  and the sum of open positions never exceeds it
  • aggressiveness 1..10 (profile): risk per trade, max position size, max
    open positions, max exposure, daily loss limit, max drawdown before the
    kill switch, minimum signal confidence and which evaluator ratings may
    trade (1-3 VIABLE; 4-7 + CONDICIONAL; 8-10 + EN_PRUEBA; never DESCARTADA)
  • mode        : 'trade' or 'close_only' (no new entries, positions run to
                  their stop-loss / take-profit)
  • allow_short : paper shorts on/off (real spot trading cannot short)

Position size = (budget x risk per trade) / stop distance, capped by the
max position size and by the budget room left. P&L is measured against the
budget: when today's loss reaches the daily limit no new entries until the next
UTC day; when the drawdown from the P&L high-water mark reaches the max
drawdown the kill switch holds entries until it is reset by hand.
Settings live in bot_metadata ('risk_settings') and can be changed from the
dashboard while the bot runs. Paper trading only.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

import config
import database as db
from strategies.base_strategy import SignalType
from utils import symbol_of

MODES = ("trade", "close_only")
_LOW = dict(risk_per_trade=0.0025, max_position=0.10, max_open=2, max_exposure=0.30,
            daily_loss=0.01, max_drawdown=0.05, min_confidence=0.60)
_HIGH = dict(risk_per_trade=0.03, max_position=0.60, max_open=8, max_exposure=1.00,
             daily_loss=0.08, max_drawdown=0.30, min_confidence=0.42)


def profile(level: int) -> dict:
    """Risk parameters for aggressiveness 1 (prudent) .. 10 (aggressive)."""
    level = min(max(int(level), 1), 10)
    t = (level - 1) / 9
    p = {k: _LOW[k] + t * (_HIGH[k] - _LOW[k]) for k in _LOW}
    p["max_open"] = int(round(p["max_open"]))
    if level <= 3:
        p["statuses"] = ("VIABLE",)
    elif level <= 7:
        p["statuses"] = ("VIABLE", "CONDICIONAL")
    else:
        p["statuses"] = ("VIABLE", "CONDICIONAL", "EN_PRUEBA")
    p["level"] = level
    return p


@dataclass
class RiskSettings:
    funds: float
    budget: float
    aggressiveness: int
    mode: str = "trade"
    allow_short: bool = True

    @classmethod
    def defaults(cls) -> "RiskSettings":
        return cls(funds=config.INITIAL_CAPITAL, budget=config.RISK_BUDGET,
                   aggressiveness=config.RISK_AGGRESSIVENESS, mode=config.RISK_MODE,
                   allow_short=config.RISK_ALLOW_SHORT)

    @classmethod
    def load(cls) -> "RiskSettings":
        raw = db.get_meta("risk_settings")
        if not raw:
            return cls.defaults()
        return cls(**{**asdict(cls.defaults()), **json.loads(raw)})

    @classmethod
    def load_and_persist(cls) -> "RiskSettings":
        """Startup: settings from the DB, or the .env defaults written to the DB so
        the dashboard and the daily review see what is actually in force."""
        s = cls.load()
        if not db.get_meta("risk_settings"):
            s.save()
        return s

    def validate(self):
        if not 1 <= int(self.aggressiveness) <= 10:
            raise ValueError("la agresividad debe estar entre 1 y 10")
        if self.funds <= 0 or self.budget <= 0:
            raise ValueError("fondos y presupuesto deben ser mayores que 0")
        if self.budget > self.funds:
            raise ValueError("el presupuesto no puede ser mayor que los fondos")
        if self.mode not in MODES:
            raise ValueError(f"modo invalido: {self.mode}")

    def save(self):
        self.validate()
        db.set_meta("risk_settings", json.dumps(asdict(self)))


class RiskEngine:

    def __init__(self, book: str, clock: Optional[Callable[[], datetime]] = None):
        self.book = book
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _key(self, name: str) -> str:
        return f"risk:{self.book}:{name}"

    # ─── State ───────────────────────────────────────────────────────────────

    def status(self, pm, prices) -> dict:
        """Current risk state of the book; updates the day reference, the P&L
        high-water mark and the kill switch. prices: {symbol: price} (every
        position is valued at its own symbol's price) or one float."""
        s = RiskSettings.load()
        prof = profile(s.aggressiveness)
        base = pm.capital_base if pm.capital_base is not None else s.budget
        positions = db.get_open_positions(book=self.book)
        exposure = sum(float(p["entry_price"]) * float(p["quantity"]) for p in positions)
        equity = sum(pm.strategy_equity(n, prices) for n in pm.strategies)
        pnl = equity - base

        today = self.clock().date().isoformat()
        day = json.loads(db.get_meta(self._key("day")) or "{}")
        if day.get("date") != today:
            day = {"date": today, "start_pnl": pnl}
            db.set_meta(self._key("day"), json.dumps(day))
        day_pnl = pnl - day["start_pnl"]
        daily_limit = prof["daily_loss"] * s.budget

        peak_raw = db.get_meta(self._key("peak_pnl"))
        peak = pnl if peak_raw == "RESET" else max(float(peak_raw or 0.0), pnl)
        db.set_meta(self._key("peak_pnl"), repr(peak))
        drawdown = peak - pnl
        max_dd = prof["max_drawdown"] * s.budget
        kill = json.loads(db.get_meta(self._key("kill")) or "null")
        if kill is None and drawdown >= max_dd:
            kill = {"at": self.clock().isoformat(),
                    "reason": f"caida de {drawdown:.2f} USD desde el maximo "
                              f"(limite {max_dd:.2f} USD = {prof['max_drawdown']:.0%} del presupuesto)"}
            db.set_meta(self._key("kill"), json.dumps(kill))

        return dict(settings=asdict(s), profile=prof, capital_base=base, equity=equity, pnl=pnl,
                    day_pnl=day_pnl, daily_limit=daily_limit, daily_limit_hit=day_pnl <= -daily_limit,
                    drawdown=drawdown, max_drawdown_usd=max_dd, kill_switch=kill is not None,
                    kill_reason=(kill or {}).get("reason", ""), exposure=exposure,
                    exposure_limit=prof["max_exposure"] * s.budget, open_positions=len(positions),
                    close_only=s.mode == "close_only" or kill is not None)

    def reset_kill_switch(self):
        """Manual re-activation: clears the kill switch and restarts the high-water mark."""
        db.set_meta(self._key("kill"), "null")
        db.set_meta(self._key("peak_pnl"), "RESET")

    def request_close_all(self):
        db.set_meta(self._key("close_all"), "1")

    def close_all_requested(self) -> bool:
        return db.get_meta(self._key("close_all")) == "1"

    def clear_close_all(self):
        db.set_meta(self._key("close_all"), "0")

    # ─── Entry check + position size ─────────────────────────────────────────

    def check_entry(self, strategy, signal, price: float, ml_confidence: float, pm) -> Tuple[bool, str, float]:
        """(allowed, reason, notional_usd) for a new position of this book.
        `price` is the entry symbol's price; the rest of the book is valued at
        the latest price the portfolio saw for each of its symbols."""
        prices = {**pm.known_prices(), symbol_of(strategy): price}
        st = self.status(pm, prices)
        s, prof = RiskSettings(**st["settings"]), st["profile"]
        if s.mode == "close_only":
            return False, "riesgo: modo solo cierre", 0.0
        if st["kill_switch"]:
            return False, f"riesgo: freno de emergencia activado ({st['kill_reason']})", 0.0
        if st["daily_limit_hit"]:
            return False, (f"riesgo: limite de perdida diaria alcanzado "
                           f"({st['day_pnl']:.2f} / -{st['daily_limit']:.2f} USD)"), 0.0
        if signal.type == SignalType.SELL and not s.allow_short:
            return False, "riesgo: cortos desactivados", 0.0
        if signal.confidence < prof["min_confidence"]:
            return False, (f"riesgo: confianza {signal.confidence:.2f} < {prof['min_confidence']:.2f} "
                           f"(agresividad {prof['level']})"), 0.0
        if ml_confidence < config.CONFIDENCE_THRESHOLD:
            return False, f"riesgo: confianza ML {ml_confidence:.2f} baja", 0.0
        if st["open_positions"] >= prof["max_open"]:
            return False, f"riesgo: maximo de {prof['max_open']} posiciones abiertas", 0.0

        stop = signal.stop_loss
        stop_pct = abs(price - stop) / price if stop else config.DEFAULT_STOP_LOSS_PCT
        stop_pct = max(stop_pct, 0.002)
        wanted = s.budget * prof["risk_per_trade"] / stop_pct
        room = st["exposure_limit"] - st["exposure"]
        cap = min(prof["max_position"] * s.budget, room)
        if room < config.MIN_ORDER_USD:
            return False, (f"riesgo: presupuesto agotado ({st['exposure']:.2f} de "
                           f"{st['exposure_limit']:.2f} USD en uso)"), 0.0
        notional = min(wanted, cap)
        if notional < config.MIN_ORDER_USD:
            return False, (f"riesgo: tamano {notional:.2f} USD menor al minimo de "
                           f"{config.MIN_ORDER_USD:.0f} USD de Binance"), 0.0
        return True, "", notional
