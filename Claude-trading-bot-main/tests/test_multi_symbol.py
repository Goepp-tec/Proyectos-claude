"""Several cryptocurrencies: one strategy instance per symbol, each with its own
candles, price, positions, rating and learned params."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import config
import database as db
from portfolio_manager import PortfolioManager
from strategies.base_strategy import Signal, SignalType

PRICES = {"BTCUSDT": 50_000.0, "ETHUSDT": 2_000.0}


def _client():
    client = Mock()
    client.place_market_order.side_effect = lambda symbol, side, qty: {"orderId": symbol}
    return client


def _strat(name, symbol):
    s = Mock()
    s.name, s.symbol, s.is_active = name, symbol, True
    return s


def _buy(price):
    return Signal(SignalType.BUY, 0.7, stop_loss=price * 0.9, take_profit=price * 1.1)


@pytest.fixture(autouse=True)
def _capital(temp_db, monkeypatch):
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 10_000.0)


# ─── Configuration and naming ──────────────────────────────────────────────────

def test_symbols_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("SYMBOLS", " ethusdt, BTCUSDT ,SOLUSDT,, ")
    assert config.parse_symbols() == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]   # primary first
    monkeypatch.delenv("SYMBOLS")
    assert config.parse_symbols() == [config.SYMBOL]


def test_one_instance_per_symbol_keeps_btc_names():
    from strategies import for_symbol
    from strategies.ema5_momentum import EMA5MomentumStrategy
    btc = for_symbol(EMA5MomentumStrategy, "BTCUSDT")
    eth = for_symbol(EMA5MomentumStrategy, "ETHUSDT")
    # BTC keeps its name: ratings, learned params and trades of earlier runs still apply
    assert (btc.name, btc.symbol) == ("EMA5_Momentum", "BTCUSDT")
    assert (eth.name, eth.symbol) == ("EMA5_Momentum@ETH", "ETHUSDT")
    assert eth.base_name == "EMA5_Momentum" and eth.coin == "ETH"
    eth.params["ema_period"] = 7
    assert btc.params["ema_period"] != 7                   # params are per instance
    c = eth.clone()
    assert (c.name, c.symbol, c.params["ema_period"]) == ("EMA5_Momentum@ETH", "ETHUSDT", 7)


def test_line_up_covers_every_symbol():
    from strategies import ALL_STRATEGIES, CANDIDATE_STRATEGIES, build_line_up
    line_up = build_line_up(ALL_STRATEGIES + CANDIDATE_STRATEGIES, ["BTCUSDT", "ETHUSDT"])
    assert len(line_up) == 2 * len(ALL_STRATEGIES + CANDIDATE_STRATEGIES)
    assert len({s.name for s in line_up}) == len(line_up)


# ─── Portfolio: orders, SL/TP and equity per symbol ───────────────────────────

def test_each_position_uses_its_own_symbol_and_price():
    client = _client()
    pm = PortfolioManager(client, [_strat("A", "BTCUSDT"), _strat("A@ETH", "ETHUSDT")])
    assert pm.process_signal(pm.strategies["A@ETH"], _buy(2_000.0), 2_000.0, 0.6)
    client.place_market_order.assert_called_with("ETHUSDT", "BUY", pytest.approx(0.0, abs=10))
    pos = db.get_open_positions("A@ETH")[0]
    assert pos["symbol"] == "ETHUSDT"

    # BTC far above ETH's take profit must not close the ETH position
    pm.check_open_positions({"BTCUSDT": 60_000.0, "ETHUSDT": 2_050.0})
    assert len(db.get_open_positions("A@ETH")) == 1
    pm.check_open_positions({"BTCUSDT": 50_000.0, "ETHUSDT": 2_300.0})     # ETH take profit
    trade = db.get_trades("A@ETH")[0]
    assert trade["symbol"] == "ETHUSDT" and trade["exit_reason"] == "TAKE_PROFIT"


def test_a_missing_price_never_closes_a_position():
    pm = PortfolioManager(_client(), [_strat("A@ETH", "ETHUSDT")])
    pm.process_signal(pm.strategies["A@ETH"], _buy(2_000.0), 2_000.0, 0.6)
    pm.check_open_positions({"BTCUSDT": 50_000.0})                         # no ETH price yet
    assert len(db.get_open_positions("A@ETH")) == 1


def test_equity_and_balance_value_each_position_at_its_symbol_price():
    pm = PortfolioManager(_client(), [_strat("A", "BTCUSDT"), _strat("A@ETH", "ETHUSDT")])
    pm.process_signal(pm.strategies["A"], _buy(50_000.0), 50_000.0, 0.6)
    pm.process_signal(pm.strategies["A@ETH"], _buy(2_000.0), 2_000.0, 0.6)
    eth = db.get_open_positions("A@ETH")[0]
    up_eth = {"BTCUSDT": 50_000.0, "ETHUSDT": 2_100.0}      # BTC unchanged since its entry
    gain = (2_100.0 - eth["entry_price"]) * eth["quantity"]
    bal = pm.total_balance(up_eth)
    assert bal["unrealized_pnl"] == pytest.approx(gain, rel=1e-3, abs=0.01)
    assert pm.strategy_equity("A@ETH", up_eth) - pm.strategy_equity("A@ETH", {"ETHUSDT": 2_000.0}) \
        == pytest.approx(100.0 * eth["quantity"])


def test_risk_engine_values_the_whole_book_at_each_symbol_price():
    from risk_engine import RiskEngine, RiskSettings
    RiskSettings(funds=1_000, budget=100, aggressiveness=5).save()
    risk = RiskEngine("main")
    pm = PortfolioManager(_client(), [_strat("A", "BTCUSDT"), _strat("A@ETH", "ETHUSDT")],
                          capital_base=100, risk_engine=risk)
    assert pm.process_signal(pm.strategies["A@ETH"], _buy(2_000.0), 2_000.0, 0.6)
    eth = db.get_open_positions("A@ETH")[0]
    flat = risk.status(pm, PRICES)["pnl"]
    down = risk.status(pm, {**PRICES, "ETHUSDT": 1_900.0})["pnl"]
    assert flat - down == pytest.approx(100.0 * eth["quantity"])
    # an entry on BTC sees the ETH position at ETH's price, not BTC's
    pm.process_signal(pm.strategies["A"], _buy(50_000.0), 50_000.0, 0.6)
    assert len(db.get_open_positions(book="main")) == 2
    # valued at BTC's price the ETH position would show ~+350 USD
    assert abs(risk.status(pm, pm.known_prices())["pnl"]) < 0.05


# ─── Bot: candles, prices and history per symbol ──────────────────────────────

def _candles(symbol, interval, n=30, start=datetime(2026, 9, 1, tzinfo=timezone.utc)):
    freq = {"1h": "1h", "4h": "4h", "1d": "1D"}[interval]
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    base = PRICES[symbol]
    close = base * (1 + np.arange(n) * 0.001)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1.0}, index=idx)


def _bot(symbols=("BTCUSDT", "ETHUSDT")):
    import main
    bot = main.TradingBot.__new__(main.TradingBot)
    bot.symbols = list(symbols)
    bot.client = Mock()
    bot.client.get_latest_candles.side_effect = lambda sym, iv, limit=600: _candles(sym, iv)
    bot.client.get_historical_klines.side_effect = lambda sym, iv, days: _candles(sym, iv, 60)
    bot.client.get_current_price.side_effect = lambda sym: PRICES[sym]
    bot._strat_dfs, bot._logged_candle, bot._prices = {}, {}, {}
    return bot


def test_candles_and_prices_are_kept_per_symbol(temp_db):
    bot = _bot()
    bot.strategies = [_strat("S", "BTCUSDT"), _strat("S@ETH", "ETHUSDT")]
    for s in bot.strategies:
        s.candle_interval = "1h"
    bot._update_price()
    bot._refresh_candles()
    assert bot.prices() == PRICES
    assert bot._current_price == PRICES["BTCUSDT"]           # primary symbol, as before
    assert bot._strat_dfs[("ETHUSDT", "1h")]["close"].iloc[0] == pytest.approx(2_000.0)
    assert bot._strat_dfs[("BTCUSDT", "1h")]["close"].iloc[0] == pytest.approx(50_000.0)


def test_each_strategy_trades_its_symbol_candles_at_its_symbol_price(temp_db):
    bot = _bot()
    seen = {}

    class Probe:
        candle_interval, min_candles, is_active = "1h", 5, True

        def __init__(self, name, symbol):
            self.name, self.symbol = name, symbol

        def generate_signal(self, df):
            seen[self.name] = float(df["close"].iloc[0])
            return Signal(SignalType.BUY, 0.7, stop_loss=df["close"].iloc[-1] * 0.9,
                          take_profit=df["close"].iloc[-1] * 1.1)
    bot.strategies = [Probe("P", "BTCUSDT"), Probe("P@ETH", "ETHUSDT")]
    bot._update_price()
    bot._refresh_candles()
    pm = Mock(book="lab")
    bot._process_signals(bot.strategies, pm, bot.prices(), lambda name, df: 0.6)
    assert seen == {"P": pytest.approx(50_000.0), "P@ETH": pytest.approx(2_000.0)}
    prices_used = {c.args[0].name: c.args[2] for c in pm.process_signal.call_args_list}
    assert prices_used == {"P": 50_000.0, "P@ETH": 2_000.0}


def test_learning_history_is_per_symbol(temp_db):
    bot = _bot()
    df = bot._learning_history("1d", 60, datetime(2027, 1, 1, tzinfo=timezone.utc), "ETHUSDT")
    assert df["close"].iloc[0] == pytest.approx(2_000.0)
    bot.client.get_historical_klines.assert_called_with("ETHUSDT", "1d", 60)


def test_evaluator_rates_each_symbol_on_its_own_history(temp_db):
    from strategy_evaluator import StrategyEvaluator
    asked = []

    def history(interval, days, end, symbol):
        asked.append((symbol, interval))
        return _candles(symbol, interval, 60)
    a, b = Mock(), Mock()
    for s, (name, sym) in ((a, ("S", "BTCUSDT")), (b, ("S@ETH", "ETHUSDT"))):
        s.name, s.symbol, s.candle_interval = name, sym, "1d"
        s.clone.return_value.min_candles = 5
        s.tunable_values.return_value = {}
    ev = StrategyEvaluator({"S": a, "S@ETH": b}, history_fn=history,
                           clock=lambda: datetime(2026, 11, 1, tzinfo=timezone.utc),
                           backtest_fn=lambda strat, df: ([], 0.0))
    ev.run_cycle_if_due()
    assert sorted(asked) == [("BTCUSDT", "1d"), ("ETHUSDT", "1d")]


def test_lab_and_baseline_get_the_same_capital_per_strategy_as_with_one_symbol():
    import main
    from strategies import ALL_STRATEGIES
    names = {S().name for S in ALL_STRATEGIES}
    one = main.build_baselines(names, ["BTCUSDT"])
    two = main.build_baselines(names, ["BTCUSDT", "ETHUSDT"])
    assert len(two) == 2 * len(one) and all(b.frozen for b in two)
    assert {b.symbol for b in two} == {"BTCUSDT", "ETHUSDT"}
    assert main.book_capital(["BTCUSDT", "ETHUSDT"]) == 2 * config.INITIAL_CAPITAL
