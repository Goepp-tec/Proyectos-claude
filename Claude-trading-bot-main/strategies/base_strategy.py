"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any

import pandas as pd


class SignalType(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    type: SignalType
    confidence: float          # 0.0 – 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.type != SignalType.HOLD


@dataclass(frozen=True)
class ParamSpec:
    """Hard limits and adjustment step for one tunable strategy parameter."""
    min: float
    max: float
    step: float

    def clamp(self, value):
        v = min(max(value, self.min), self.max)
        if all(float(x).is_integer() for x in (self.min, self.max, self.step)):
            return int(round(v))
        return round(float(v), 6)


class BaseStrategy(ABC):
    """
    All concrete strategies inherit from this class.
    Each strategy must implement `generate_signal` and declare
    its required indicator lookback via `min_candles`.

    TUNABLE_PARAMS declares the only params the learning engine may change,
    with hard min / max limits and the size of one adjustment step.
    """

    TUNABLE_PARAMS: Dict[str, ParamSpec] = {}
    SOURCE: str = ""   # where a catalog strategy comes from (book, author, public setup)

    def __init__(self, name: str, params: Dict[str, Any]):
        import config
        self.name = name
        self.params = dict(params)
        # Symbol this instance trades; strategies.for_symbol() makes one per coin
        self.symbol: str = config.SYMBOL
        self.capital: float = 0.0
        self.is_active: bool = False
        self.frozen: bool = False   # frozen = baseline copy, params never change

        # Running counters – updated by PortfolioManager
        self.total_trades: int = 0
        self.winning_trades: int = 0

    # ─── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Analyse the most recent candles and return a Signal.
        `df` always contains at least `min_candles` rows with all indicators
        pre-computed by `utils.indicators.add_all_indicators`.
        """

    @property
    @abstractmethod
    def min_candles(self) -> int:
        """Minimum number of candles needed for reliable signals."""

    @property
    @abstractmethod
    def candle_interval(self) -> str:
        """Binance interval string, e.g. '1h', '4h', '1d'."""

    @property
    def max_hold_candles(self) -> int:
        """
        Maximum number of candles to hold a position before forced exit.
        Default is 48 (2 days on 1h, 8 days on 4h, 48 days on 1d).
        Override in daily strategies that need to capture longer trends.
        """
        return 48

    # ─── Convenience helpers ──────────────────────────────────────────────────

    def update_params(self, new_params: Dict[str, Any]):
        if self.frozen:
            raise RuntimeError(f"{self.name} is frozen (baseline); params cannot change")
        self.params.update(new_params)

    # ─── Tunable params (learning engine) ─────────────────────────────────────

    def freeze(self):
        self.frozen = True

    def set_tunable_param(self, name: str, value):
        """Set a declared tunable param, clamped to its hard limits. Returns the value set."""
        if self.frozen:
            raise RuntimeError(f"{self.name} is frozen (baseline); params cannot change")
        spec = self.TUNABLE_PARAMS[name]   # KeyError if not declared
        self.params[name] = spec.clamp(value)
        return self.params[name]

    def tunable_values(self) -> Dict[str, Any]:
        return {k: self.params[k] for k in self.TUNABLE_PARAMS}

    def restore_tunables(self, saved: Dict[str, Any]):
        """Re-apply persisted learned values; undeclared keys are ignored."""
        for name, value in (saved or {}).items():
            if name in self.TUNABLE_PARAMS:
                self.set_tunable_param(name, value)

    def clone(self, params: Optional[Dict[str, Any]] = None) -> "BaseStrategy":
        """Fresh instance of the same strategy, symbol and name with these params (never frozen)."""
        other = type(self)()
        other.name, other.symbol = self.name, self.symbol
        other.params = dict(self.params)
        if params:
            other.params.update(params)
        return other

    @property
    def base_name(self) -> str:
        """Strategy name without the coin suffix ('EMA5_Momentum@ETH' -> 'EMA5_Momentum')."""
        return self.name.split("@")[0]

    @property
    def coin(self) -> str:
        from utils import coin_of
        return coin_of(self.symbol)

    def set_capital(self, capital: float):
        self.capital = capital

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def record_trade_outcome(self, won: bool):
        self.total_trades += 1
        if won:
            self.winning_trades += 1

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} capital={self.capital:.2f} active={self.is_active}>"
