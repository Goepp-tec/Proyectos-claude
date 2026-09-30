"""
BTC Paper Trading Bot — Main Orchestrator
─────────────────────────────────────────
Market data: REAL Binance public API (no keys required)
Execution:   Paper trading at live prices by default

Startup sequence:
  1. Verify Binance API connectivity (public endpoints)
  2. Fetch 90 days of real OHLCV data for backtesting
  3. Backtest all 5 strategies on real historical data
  4. Activate strategies that pass WR + PF thresholds
  5. Launch trading loops:
       - Trading loop: signal generation every 60s
       - Position loop: SL/TP monitoring every 20s
       - Learning loop: journal + param tuning every 3min
       - Balance loop: equity snapshots every 5min
  6. Start Dash dashboard on port 8050

Changes:
  - Added log rotation to prevent unbounded log growth
"""

import json
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List

import config
import database as db
from binance_client import BinanceClient
from backtester import run_all_backtests
from portfolio_manager import PortfolioManager
from learning_engine import LearningEngine
from adaptive_tuner import AdaptiveTuner
from strategies import ALL_STRATEGIES, CANDIDATE_STRATEGIES, build_line_up
from strategy_evaluator import StrategyEvaluator
from risk_engine import RiskEngine, RiskSettings, profile as risk_profile
from market_data import MarketDataCollector, enrich as enrich_market_data
from utils import price_of, symbol_of
from daily_report import DailyReporter
from confirmations import ConfirmationEngine
from news_data import NewsCollector

# ─── Logging setup ────────────────────────────────────────────────────────────
# Console handler with colors (if available)
try:
    import colorlog
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan", "INFO": "green",
            "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold_red",
        },
    ))
except ImportError:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"))

# File handler with rotation (max 5MB per file, keep 3 backup files)
log_dir = os.path.dirname(config.LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    config.LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,
    encoding="utf-8",
)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
root_logger.handlers = [console_handler, file_handler]

logger = logging.getLogger("main")

# ─── Trading mode ─────────────────────────────────────────────────────────────

def select_trading_mode(strategies, bt_results, allow_unvalidated: bool):
    """
    Returns (active_strategies, book, mode):
      TRADE             – only strategies that passed the backtest, book 'main'
      TRADE_UNVALIDATED – none passed but ALLOW_UNVALIDATED_STRATEGIES: all, 'main'
      OBSERVE           – none passed: all strategies run in the 'observe' book
                          (signals + theoretical equity, no positions/orders)
    """
    passing = [s for s in strategies
               if bt_results.get(s.name) is not None and bt_results[s.name].passes_threshold]
    if passing:
        return passing, "main", "TRADE"
    if allow_unvalidated:
        return list(strategies), "main", "TRADE_UNVALIDATED"
    return list(strategies), "observe", "OBSERVE"


def resolve_trading_mode(strategies, bt_results, allow_unvalidated: bool, lock: bool = False):
    """
    select_trading_mode(), optionally locked: with lock=True the mode and active
    set chosen at the first start are stored and reused on every restart, so a
    long test run keeps comparing the same books even if the startup backtest
    (re-run on each restart over a shifted 500-day window) changes its verdict.
    """
    if lock:
        saved = db.get_meta("trading_mode_lock")
        if saved:
            s = json.loads(saved)
            if s["mode"] in ("OBSERVE", "TRADE_UNVALIDATED"):
                active = list(strategies)   # "all strategies", incl. ones added since
            else:
                active = [x for x in strategies if x.name in set(s["active"])]
            logger.info(f"Trading mode locked by the running test: {s['mode']} ({s['book']})")
            return active, s["book"], s["mode"]
    active, book, mode = select_trading_mode(strategies, bt_results, allow_unvalidated)
    if lock:
        db.set_meta("trading_mode_lock", json.dumps(
            {"mode": mode, "book": book, "active": [x.name for x in active]}))
    return active, book, mode


def make_portfolio(client, strategies, book: str, clock=None,
                   capital_base=None, risk_engine=None) -> PortfolioManager:
    """Only the 'main' book sends orders through the client (paper or live)."""
    return PortfolioManager(client, strategies, book=book,
                            simulate_fills=(book != "main"), clock=clock,
                            capital_base=capital_base, risk_engine=risk_engine)


def restore_learned_params(learners):
    AdaptiveTuner.restore_learned_params({s.name: s for s in learners})


def build_lab(learners):
    """
    'lab' copies of the learning strategies: they share the params dict (tuned
    values follow) but keep their own signal state, and trade every signal
    without the evaluator gate — the live evidence the evaluator reads.
    """
    lab = []
    for s in learners:
        copy = type(s)()
        copy.name, copy.symbol = s.name, symbol_of(s)
        copy.params = s.params
        copy.is_active = True
        lab.append(copy)
    return lab


def build_baselines(active_names, symbols=None):
    """Frozen default-param copy of every registered strategy on every symbol;
    active if its learner is active."""
    baselines = []
    for b in build_line_up(ALL_STRATEGIES, symbols or config.SYMBOLS):
        b.is_active = b.name in active_names
        b.freeze()
        baselines.append(b)
    return baselines


def book_capital(symbols=None) -> float:
    """
    Capital of the lab and baseline books: INITIAL_CAPITAL per symbol, so each
    strategy instance keeps the same share it had with one symbol (1000 USD over
    25 strategies x 5 coins would leave 8 USD each, below Binance's minimum).
    Their returns are compared in %, never in dollars.
    """
    return config.INITIAL_CAPITAL * len(symbols or config.SYMBOLS)


# ─── Shutdown flag ────────────────────────────────────────────────────────────
_shutdown = threading.Event()

def _handle_signal(sig, frame):
    logger.warning(f"Signal {sig} received — shutting down gracefully…")
    _shutdown.set()

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


class TradingBot:

    def __init__(self):
        db.init_db()
        logger.info("Database initialised")
        self.client     = BinanceClient()
        # Every symbol runs its own copy of every strategy: registered ones + the
        # candidate catalog; the strategy evaluator decides which of them may
        # trade in the learning book, and when (per symbol).
        self.symbols    = list(config.SYMBOLS)
        self.strategies = build_line_up(ALL_STRATEGIES + CANDIDATE_STRATEGIES, self.symbols)
        self.portfolio:  PortfolioManager = None
        self.learning:   LearningEngine   = None
        self.book = "main"      # book the (learning) strategies trade in
        self.mode = "TRADE"
        self.lock_mode = False  # test runs lock the first-start mode across restarts
        self.baselines = []     # frozen default-param copies (book 'baseline')
        self.baseline_portfolio: PortfolioManager = None
        self.tuner: AdaptiveTuner = None
        self.lab = []           # ungated copies of all learners (book 'lab')
        self.lab_portfolio: PortfolioManager = None
        self.evaluator: StrategyEvaluator = None
        self.risk: RiskEngine = None   # budget / aggressiveness for the learning book
        self.reporter: DailyReporter = None   # daily report (reports/informe_diario_*.txt)
        # Free positioning / funding / Fear & Greed data, joined to every candle
        self.market = MarketDataCollector(self.symbols)
        self.news = NewsCollector(self.symbols)       # headlines, macro calendar, stablecoins
        # Extra confirmations for the learning book (technical, sentiment, news, Claude)
        self.confirm = ConfirmationEngine(dfs_fn=lambda sym, iv: self._strat_dfs.get((sym, iv)),
                                          symbols=self.symbols)
        self._strat_dfs: Dict[tuple, object] = {}     # (symbol, interval) -> candles
        self._logged_candle: Dict[str, str] = {}
        self._prices: Dict[str, float] = {}           # symbol -> latest price
        self._current_price: float = 0.0              # primary symbol (BTC)
        self._price_lock = threading.Lock()

    # ─── Startup ──────────────────────────────────────────────────────────────

    def startup(self):
        logger.info("=" * 65)
        logger.info("   BTC PAPER TRADING BOT  —  Starting up")
        logger.info(f"   Mode      : {'PAPER TRADING (real Binance data)' if self.client.is_paper_trading else 'LIVE TRADING ⚠️'}")
        logger.info(f"   Capital   : ${config.INITIAL_CAPITAL:,.2f}")
        logger.info(f"   Symbols   : {', '.join(self.symbols)}")
        logger.info(f"   Backtest  : {config.BACKTEST_DAYS} days of real OHLCV data")
        logger.info("=" * 65)

        # Fetch live prices
        self._update_price()
        stats = self.client.get_24hr_stats(config.SYMBOL)
        change_pct = float(stats.get("priceChangePercent", 0))
        logger.info(
            f"Live BTC price: ${self._current_price:,.2f} "
            f"({change_pct:+.2f}% 24h | vol: {float(stats.get('volume', 0)):,.0f} BTC)"
        )
        for sym in self.symbols[1:]:
            logger.info(f"Live {sym} price: ${self._prices.get(sym, 0):,.4f}")

        # Warm up candle cache
        logger.info("\nLoading candle data…")
        self._refresh_candles()

        # Learned parameter values from previous runs (declared tunables only)
        restore_learned_params(self.strategies)

        # Backtest on real data
        logger.info(f"\nRunning backtests on {config.BACKTEST_DAYS} days of real Binance data…")
        bt_results = run_all_backtests(self.strategies, self.client)

        for strat in self.strategies:
            result = bt_results.get(strat.name)
            if result:
                symbol, level = ("✓", "info") if result.passes_threshold else ("✗", "warning")
                getattr(logger, level)(
                    f"  {symbol} {strat.name:20s}  CAGR={result.cagr*100:.1f}% "
                    f"WR={result.win_rate*100:.1f}% PF={result.pf_display} "
                    f"trades={result.total_trades}"
                )
            else:
                logger.warning(f"  ✗ {strat.name:20s}  no backtest result")

        # Decide what trades: validated strategies, all (only if explicitly
        # allowed) or nothing — observation mode.
        active, self.book, self.mode = resolve_trading_mode(
            self.strategies, bt_results, config.ALLOW_UNVALIDATED_STRATEGIES,
            lock=self.lock_mode,
        )
        active_names = {s.name for s in active}
        trades_main = self.book == "main"
        share = config.INITIAL_CAPITAL / max(len(active), 1)
        for strat in self.strategies:
            strat.is_active = strat.name in active_names
            result = bt_results.get(strat.name)
            in_main = trades_main and strat.is_active
            # Preserve existing capital on restart to avoid double-counting with
            # any open positions whose notional was already deducted in a prior run.
            existing_row = db.get_strategy(strat.name)
            if in_main:
                new_capital = existing_row["capital"] if (existing_row and existing_row["capital"] > 0) else share
            else:
                new_capital = 0
            db.upsert_strategy(
                name=strat.name,
                capital=new_capital,
                params=strat.params,
                backtest_cagr=result.cagr if result else 0,
                backtest_win_rate=result.win_rate if result else 0,
                is_active=in_main,
            )
        db.set_meta("trading_mode", self.mode)

        if self.mode == "OBSERVE":
            logger.warning(
                "No strategy passed backtest thresholds → OBSERVATION MODE: signals and "
                "theoretical equity are recorded in the 'observe' book, no positions are "
                "opened. Set ALLOW_UNVALIDATED_STRATEGIES=true to trade them anyway."
            )
        elif self.mode == "TRADE_UNVALIDATED":
            logger.warning(
                "No strategy passed backtest thresholds, but ALLOW_UNVALIDATED_STRATEGIES=true "
                "→ trading ALL strategies (unvalidated)."
            )
        logger.info(f"\nMode {self.mode}: {len(active)}/{len(self.strategies)} strategies "
                    f"({len(self.symbols)} symbols) in book '{self.book}'\n")

        # Init portfolio + learning. The learning book trades the risk budget,
        # sized and limited by the risk engine (aggressiveness 1-10).
        risk_settings = RiskSettings.load_and_persist()
        self.risk = RiskEngine(self.book)
        self.portfolio = make_portfolio(self.client, self.strategies, self.book,
                                        capital_base=risk_settings.budget, risk_engine=self.risk)
        prof = risk_profile(risk_settings.aggressiveness)
        logger.info(
            f"Risk: funds ${risk_settings.funds:,.0f}, budget ${risk_settings.budget:,.0f}, "
            f"aggressiveness {prof['level']}/10 (risk/trade {prof['risk_per_trade']:.2%}, "
            f"daily loss limit {prof['daily_loss']:.1%}, kill switch at {prof['max_drawdown']:.0%} "
            f"drawdown, ratings allowed: {', '.join(prof['statuses'])}), mode {risk_settings.mode}"
        )
        strat_dict = {s.name: s for s in self.strategies}
        self.learning = LearningEngine(strat_dict)

        # Frozen baselines trade the same signals with default params, own capital
        self.baselines = build_baselines(active_names, self.symbols)
        self.baseline_portfolio = make_portfolio(self.client, self.baselines, "baseline",
                                                 capital_base=book_capital(self.symbols))
        # Parameter tuning is compared against baselines, so it only covers the
        # registered strategies (candidates are rated by the evaluator instead).
        registered = {S().name for S in ALL_STRATEGIES}
        self.tuner = AdaptiveTuner(
            learners={n: s for n, s in strat_dict.items() if s.base_name in registered},
            history_fn=self._learning_history,
            equity_fn=self._strategy_equity,
            clock=lambda: datetime.now(timezone.utc),
            learner_book=self.book,
        )
        # Lab copies trade everything not discarded: live evidence for the evaluator
        self.lab = build_lab(self.strategies)
        self.lab_portfolio = make_portfolio(self.client, self.lab, "lab",
                                            capital_base=book_capital(self.symbols))
        self.evaluator = StrategyEvaluator(
            strat_dict, history_fn=self._learning_history,
            clock=lambda: datetime.now(timezone.utc), live_book="lab",
        )
        logger.info(
            f"Learning: {'ON' if config.LEARNING_ENABLED else 'OFF'} "
            f"(every {config.LEARNING_INTERVAL_HOURS:g}h, walk-forward "
            f"{config.LEARNING_PROPOSAL_DAYS}d proposal / {config.LEARNING_VALIDATION_DAYS}d validation); "
            f"{sum(b.is_active for b in self.baselines)} frozen baseline copies; "
            f"evaluator {'ON' if config.EVAL_ENABLED else 'OFF'} over {len(self.strategies)} strategies "
            f"({len(CANDIDATE_STRATEGIES)} from the candidate catalog)"
        )

        # Load journal entries and restore learned patterns from previous runs
        self.learning.learn_from_all_journal_entries()
        # Recent closed trades drive the ML confidence; rebuild them after a restart
        self.learning.seed_from_trades(db.get_trades(limit=5000, book=self.book))

        # Initial balance snapshot
        bal = self.portfolio.total_balance(self.prices())
        db.record_balance(
            total_balance=bal["total_balance"],
            realized_pnl=bal["realized_pnl"],
            unrealized_pnl=bal["unrealized_pnl"],
            strategy_breakdown=bal.get("breakdown", {}),
            book=self.book,
        )

        self.reporter = self._build_reporter()
        logger.info("Startup complete. Entering trading loops.\n")

    # ─── Main run ─────────────────────────────────────────────────────────────

    def run(self):
        self.startup()

        threads = [
            threading.Thread(target=self._trading_loop,    daemon=True, name="trading"),
            threading.Thread(target=self._position_loop,   daemon=True, name="positions"),
            threading.Thread(target=self._learning_loop,   daemon=True, name="learning"),
            threading.Thread(target=self._balance_loop,    daemon=True, name="balance"),
            threading.Thread(target=self._dashboard_thread, daemon=True, name="dashboard"),
        ]
        for t in threads:
            t.start()
            logger.info(f"Thread started: {t.name}")

        logger.info(f"\n🚀 Bot running. Dashboard: http://localhost:{config.DASHBOARD_PORT}\n")

        while not _shutdown.is_set():
            _shutdown.wait(timeout=5)

        logger.info("Shutdown complete.")

    # ─── Trading loop ─────────────────────────────────────────────────────────

    def _trading_loop(self):
        logger.info("[trading] Loop started")
        while not _shutdown.is_set():
            try:
                self._update_price()
                self._refresh_candles()
                price = self.prices()

                self._process_signals(self.strategies, self.portfolio, price,
                                      lambda name, df: self.learning.get_confidence(name, df),
                                      gate=self._learner_gate)
                self._process_signals(self.baselines, self.baseline_portfolio, price,
                                      lambda name, df: config.BASELINE_ML_CONFIDENCE)
                self._process_signals(self.lab, self.lab_portfolio, price,
                                      lambda name, df: config.BASELINE_ML_CONFIDENCE,
                                      gate=self._lab_gate)

            except Exception as e:
                logger.error(f"[trading] Error: {e}", exc_info=True)

            _shutdown.wait(timeout=config.STRATEGY_CHECK_INTERVAL_SEC)

    def _learner_gate(self, strat, signal, df):
        """Blocked reason for the learning book, or None: first the evaluator (is
        the strategy viable here?), then the confirmation engine (do independent
        checks agree now?). The checklist is kept in the signal's metadata."""
        prof = risk_profile(RiskSettings.load().aggressiveness)
        if config.EVAL_ENABLED and getattr(self, "evaluator", None) is not None:
            ok, why = self.evaluator.can_trade(strat, signal.type, df, prof["statuses"])
            if not ok:
                return why
        confirm = getattr(self, "confirm", None)
        if config.CONFIRMATIONS_ENABLED and confirm is not None:
            ok, why, summary = confirm.decide(confirm.checks(strat, signal), prof["min_confirmations"])
            signal.metadata = {**(signal.metadata or {}), "confirmaciones": summary}
            if not ok:
                return why
        return None

    def _risk_housekeeping(self, price):
        """Apply dashboard risk changes and refresh the day / kill-switch state.
        price: {symbol: price} (or one float with a single symbol)."""
        if self.risk is None or self.portfolio is None or not price:
            return
        settings = RiskSettings.load()
        if self.portfolio.capital_base != settings.budget:
            logger.info(f"[risk] budget changed to ${settings.budget:,.2f}")
            self.portfolio.set_capital_base(settings.budget, price)
        if self.risk.close_all_requested():
            n = self.portfolio.close_all_positions(price, reason="MANUAL_CLOSE_ALL")
            settings.mode = "close_only"
            settings.save()
            self.risk.clear_close_all()
            logger.warning(f"[risk] close-all requested: {n} positions closed, mode -> close_only")
        st = self.risk.status(self.portfolio, price)
        if st["kill_switch"] and not getattr(self, "_kill_logged", False):
            logger.warning(f"[risk] KILL SWITCH: {st['kill_reason']} — no new entries until reset")
        self._kill_logged = st["kill_switch"]

    def _lab_gate(self, strat, signal, df):
        """The lab tests everything except discarded strategies (value 0, never again)."""
        if StrategyEvaluator.is_discarded(strat.name):
            return "evaluator: DESCARTADA (valor 0)"
        return None

    def _process_signals(self, strategies, portfolio, prices, ml_conf_fn, gate=None):
        """Each strategy reads its own symbol's candles and trades at its price."""
        book = portfolio.book
        for strat in strategies:
            if not strat.is_active:
                continue
            interval, symbol = strat.candle_interval, symbol_of(strat)
            price = price_of(prices, symbol)
            if price <= 0:
                continue
            df = self._strat_dfs.get((symbol, interval))
            if df is None or len(df) < strat.min_candles:
                logger.debug(
                    f"[{book}:{strat.name}] Insufficient data "
                    f"({len(df) if df is not None else 0}/{strat.min_candles} candles)"
                )
                continue

            ml_conf = ml_conf_fn(strat.name, df)
            signal = strat.generate_signal(df)

            if signal.is_actionable:
                candle = str(df.index[-1])
                if self._logged_candle.get((book, strat.name)) != candle:
                    self._logged_candle[(book, strat.name)] = candle
                    logger.info(
                        f"[{book}:{strat.name}] SIGNAL {signal.type.value} "
                        f"conf={signal.confidence:.2f} ml={ml_conf:.2f} "
                        f"price=${price:,.2f} candle={candle}"
                    )
                placed = portfolio.process_signal(
                    strat, signal, price, ml_confidence=ml_conf,
                    candle_ts=candle,
                    blocked_reason=gate(strat, signal, df) if gate else None,
                )
                if placed:
                    logger.info(f"[{book}:{strat.name}] ✓ Paper trade opened")
            else:
                logger.debug(f"[{book}:{strat.name}] HOLD")

    # ─── Position monitoring loop (SL/TP) ─────────────────────────────────────

    def _position_loop(self):
        logger.info("[positions] Loop started")
        while not _shutdown.is_set():
            try:
                price = self.prices()
                if price and self.portfolio:
                    self.portfolio.check_open_positions(price)
                    self.baseline_portfolio.check_open_positions(price)
                    self.lab_portfolio.check_open_positions(price)
                    self._risk_housekeeping(price)
            except Exception as e:
                logger.error(f"[positions] Error: {e}", exc_info=True)
            _shutdown.wait(timeout=config.POSITION_CHECK_INTERVAL_SEC)

    # ─── Learning loop ────────────────────────────────────────────────────────

    def _learning_loop(self):
        logger.info("[learning] Loop started")
        while not _shutdown.is_set():
            try:
                self._journal_new_trades()

                strat_dict = {s.name: s for s in self.strategies}
                self.learning.update_performance_snapshots(strat_dict)

                self.market.update()          # throttled to MARKET_DATA_INTERVAL_MIN
                self.news.update()            # headlines hourly, calendar 6 h, stablecoins daily
                if config.EVAL_ENABLED and self.evaluator:
                    self.evaluator.run_cycle_if_due()
                if config.LEARNING_ENABLED and self.tuner:
                    self.tuner.run_cycle_if_due()
                self._write_daily_report()     # once per UTC day

            except Exception as e:
                logger.error(f"[learning] Error: {e}", exc_info=True)
            _shutdown.wait(timeout=config.LEARNING_UPDATE_INTERVAL_SEC)

    def _journal_new_trades(self):
        """
        Feed each newly closed trade of the learning book to the journal once.
        The last processed trade id is persisted: the old in-memory counter
        started at 0 on every restart (re-journaling all trades) and stopped
        seeing new trades once 1000 existed.
        """
        key = f"journal:last_trade_id:{self.book}"
        last_id = int(db.get_meta(key) or 0)
        new_trades = sorted((t for t in db.get_trades(limit=100_000, book=self.book)
                             if t["id"] > last_id), key=lambda t: t["id"])
        for trade in new_trades:
            strat     = self._get_strat(trade["strategy_name"])
            interval  = strat.candle_interval if strat else "1h"
            symbol    = symbol_of(strat) if strat else (trade.get("symbol") or config.SYMBOL)
            df_latest = self._strat_dfs.get((symbol, interval))
            self.learning.on_trade_closed(
                trade_id=trade["id"],
                strategy_name=trade["strategy_name"],
                entry_price=float(trade["entry_price"]),
                exit_price=float(trade["exit_price"]),
                pnl=float(trade["pnl"]),
                pnl_pct=float(trade["pnl_pct"]),
                side=trade["side"],
                duration_hours=float(trade["duration_hours"]),
                exit_reason=trade.get("exit_reason", ""),
                entry_features=trade.get("entry_features", {}),
                df=df_latest,
                closed_at=trade.get("closed_at"),
            )
            db.set_meta(key, str(trade["id"]))

    # ─── Balance snapshot loop ────────────────────────────────────────────────

    def _balance_loop(self):
        logger.info("[balance] Loop started")
        while not _shutdown.is_set():
            try:
                self._record_balances()
            except Exception as e:
                logger.error(f"[balance] Error: {e}", exc_info=True)
            _shutdown.wait(timeout=60)  # Update balance every 60 seconds (not every 5 min)

    def _record_balances(self):
        """Equity snapshot of the learning, baseline and lab books (dashboard curves)."""
        price = self.prices()
        if not price or not self.portfolio:
            return
        bals = {}
        for book, pm in ((self.book, self.portfolio), ("baseline", self.baseline_portfolio),
                         ("lab", getattr(self, "lab_portfolio", None))):
            if pm is None:
                continue
            bal = pm.total_balance(price)
            db.record_balance(
                total_balance=bal["total_balance"],
                realized_pnl=bal["realized_pnl"],
                unrealized_pnl=bal["unrealized_pnl"],
                strategy_breakdown=bal.get("breakdown", {}),
                book=book,
            )
            bals[book] = bal
        learn = bals[self.book]
        logger.info(
            f"[balance:{self.book}] ${learn['total_balance']:,.2f} | "
            f"Realized: ${learn['realized_pnl']:+,.2f} | "
            f"Unrealized: ${learn['unrealized_pnl']:+,.2f} || "
            + " | ".join(f"{b} ${v['total_balance']:,.2f}" for b, v in bals.items() if b != self.book)
        )

    # ─── Dashboard ────────────────────────────────────────────────────────────

    def _dashboard_thread(self):
        logger.info(f"[dashboard] Starting on http://localhost:{config.DASHBOARD_PORT}")
        try:
            from dashboard.app import run_dashboard
            run_dashboard(debug=False)
        except Exception as e:
            logger.error(f"[dashboard] Failed: {e}", exc_info=True)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _update_price(self):
        """Latest price of every symbol; one failing symbol keeps its last price."""
        lock = getattr(self, "_price_lock", None) or threading.Lock()
        for sym in getattr(self, "symbols", None) or [config.SYMBOL]:
            try:
                price = self.client.get_current_price(sym)
                if price > 0:
                    with lock:
                        self._prices[sym] = price
                        if sym == config.SYMBOL:
                            self._current_price = price
            except Exception as e:
                logger.warning(f"Price update failed for {sym}: {e}")

    def prices(self) -> Dict[str, float]:
        """{symbol: latest price}; bots built before several symbols fall back to BTC."""
        prices = getattr(self, "_prices", None)
        if prices:
            return dict(prices)
        btc = getattr(self, "_current_price", 0.0)
        return {config.SYMBOL: btc} if btc else {}

    def _refresh_candles(self):
        """Refresh OHLCV data for each (symbol, interval) an active strategy uses."""
        pairs = {(symbol_of(s), s.candle_interval) for s in self.strategies if s.is_active}
        for symbol, interval in sorted(pairs):
            try:
                df = self.client.get_latest_candles(
                    symbol, interval, limit=config.LOOKBACK_CANDLES
                )
                if df is not None and not df.empty:
                    self._strat_dfs[(symbol, interval)] = enrich_market_data(df, symbol, interval)
                    logger.debug(f"Refreshed {symbol} {interval} candles: {len(df)} rows")
            except Exception as e:
                logger.error(f"[candles] Error refreshing {symbol} {interval}: {e}")

    def _strategy_equity(self, book: str, name: str) -> float:
        pm = self.baseline_portfolio if book == "baseline" else self.portfolio
        return pm.strategy_equity(name, self.prices())

    def _learning_history(self, interval: str, days: int, end: datetime, symbol: str = None):
        symbol = symbol or config.SYMBOL
        df = self.client.get_historical_klines(symbol, interval, days)
        return enrich_market_data(df[df.index < end], symbol, interval)

    def _build_reporter(self) -> DailyReporter:
        return DailyReporter(
            out_dir=os.path.join(config.DATA_DIR, "reports"),
            books={"aprende": (self.book, self.portfolio),
                   "lab": ("lab", self.lab_portfolio),
                   "baseline": ("baseline", self.baseline_portfolio)},
            prices_fn=self.prices, closes_fn=self._daily_closes,
        )

    def _daily_closes(self, symbol: str):
        return self.client.get_latest_candles(symbol, "1d", limit=10)["close"]

    def _write_daily_report(self):
        """Daily report of the previous UTC day, once; never breaks the loop."""
        if getattr(self, "reporter", None) is None:
            return None
        try:
            return self.reporter.write_if_due()
        except Exception as e:
            logger.error(f"[report] daily report failed: {e}", exc_info=True)
            return None

    def _get_strat(self, name: str):
        for s in self.strategies:
            if s.name == name:
                return s
        return None


# ─── Entry point ──────────────────────────────────────────────────────────────

def prepare_database():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        example = os.path.join(os.path.dirname(__file__), ".env.example")
        if os.path.exists(example):
            import shutil
            shutil.copy(example, env_path)
            logger.info("Created .env from .env.example")

    # Schema must exist before bot_metadata is touched (fresh/empty DB file).
    db.init_db()

    # ─── Handle data reset if configured ───────────────────────────────────────
    if config.RESET_ON_STARTUP:
        logger.warning("RESET_ON_STARTUP is enabled - clearing all trading data!")
        db.clear_old_data()
        logger.info("All trading data cleared. Starting fresh.")
    else:
        # Set live_since if not already set
        db.set_live_since()
        live_since = db.get_live_since()
        if live_since:
            logger.info(f"Live trading since: {live_since}")


def main():
    prepare_database()
    try:
        bot = TradingBot()
        bot.run()
    except ConnectionError as e:
        logger.critical(f"Cannot connect to Binance: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
