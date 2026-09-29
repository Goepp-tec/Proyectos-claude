"""Central configuration for the BTC Paper Trading Bot."""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Credentials (optional — only needed for live/testnet execution) ───────
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET        = os.getenv("USE_TESTNET", "true").lower() == "true"

# Testnet base URLs (only used if API keys are set and USE_TESTNET=true)
TESTNET_REST_URL = "https://testnet.binance.vision/api"
TESTNET_WS_URL   = "wss://testnet.binance.vision/ws"

# ─── Trading Mode ──────────────────────────────────────────────────────────────
# PAPER_TRADING=true  → simulate orders at real Binance prices (default, safe)
# PAPER_TRADING=false → live/testnet order execution (requires API keys)
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

# If NO strategy passes the startup backtest the bot runs in OBSERVATION mode:
# it logs signals and tracks theoretical equity in a separate 'observe' book but
# opens no positions and places no orders. Set to "true" to trade all
# (unvalidated) strategies anyway — the original behaviour of this bot.
ALLOW_UNVALIDATED_STRATEGIES = os.getenv("ALLOW_UNVALIDATED_STRATEGIES", "false").lower() == "true"

# ─── Trading Parameters ────────────────────────────────────────────────────────
SYMBOL             = "BTCUSDT"   # primary symbol: its strategies keep their plain names


def parse_symbols() -> list:
    """SYMBOLS=BTCUSDT,ETHUSDT,... (default: SYMBOL only). The primary symbol
    goes first; every other one runs its own copy of every strategy."""
    raw = [s.strip().upper() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]
    rest = [s for i, s in enumerate(raw) if s != SYMBOL and s not in raw[:i]]
    return [SYMBOL] + rest


SYMBOLS            = parse_symbols()
INITIAL_CAPITAL    = float(os.getenv("INITIAL_CAPITAL", "10000"))
MAX_STRATEGIES     = 7
CANDLE_INTERVAL    = "1h"
LOOKBACK_CANDLES   = 600   # candles kept in memory per strategy interval

# ─── Risk engine (risk_engine.py) — defaults; editable from the dashboard ─────
# INITIAL_CAPITAL is the paper account ("funds"). RISK_BUDGET is the most the
# bot may use (default 10% of the funds, e.g. 100 of 1000).
# RISK_AGGRESSIVENESS 1 (prudent) .. 10 (aggressive).
RISK_BUDGET         = float(os.getenv("RISK_BUDGET") or 0) or INITIAL_CAPITAL * 0.10
RISK_AGGRESSIVENESS = int(os.getenv("RISK_AGGRESSIVENESS", "5"))
RISK_MODE           = os.getenv("RISK_MODE", "trade")          # trade | close_only
RISK_ALLOW_SHORT    = os.getenv("RISK_ALLOW_SHORT", "true").lower() == "true"
MIN_ORDER_USD       = 5.0     # Binance BTCUSDT minimum notional

# ─── Risk Management ───────────────────────────────────────────────────────────
DEFAULT_STOP_LOSS_PCT              = 0.025   # 2.5%
DEFAULT_TAKE_PROFIT_PCT            = 0.055   # 5.5%
MAX_POSITION_PCT                   = 0.35    # max 35% of strategy capital per trade
MIN_POSITION_PCT                   = 0.05    # min 5% (ensures meaningful trade size)
MAX_OPEN_POSITIONS_PER_STRATEGY    = 2
MAX_PORTFOLIO_DRAWDOWN_PCT         = 0.20    # pause new entries at 20% drawdown

# ─── Fees & Slippage ───────────────────────────────────────────────────────────
TRADING_FEE = 0.001   # 0.1% Binance spot fee
SLIPPAGE    = 0.0003  # 0.03% estimated slippage (conservative)

# ─── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_DAYS          = 500   # 500 days – needed for SMA-250 on daily strategies
MIN_CAGR_THRESHOLD     = 0.30  # Require ≥30% annualised CAGR to activate a strategy
MIN_WIN_RATE           = 0.38  # 38% minimum – momentum strategies have lower WR but high R:R
MIN_PROFIT_FACTOR      = 1.20  # Min gross profit / gross loss ratio
# Below this many trades the profit factor is reported as not reliable ("n/a")
# and the strategy cannot be activated, whatever its CAGR / WR / PF.
MIN_BACKTEST_TRADES    = int(os.getenv("MIN_BACKTEST_TRADES", "30"))
PF_NO_LOSS_CAP         = 99.99  # PF reported when there are wins but no losses

# ─── Learning Engine ───────────────────────────────────────────────────────────
MIN_TRADES_FOR_LEARNING = 10    # start ML tuning after N trades
MODEL_UPDATE_FREQUENCY  = 5     # retrain model every N closed trades
CONFIDENCE_THRESHOLD    = 0.40  # skip trades with ML confidence below this
# Confidence = recent win rate shrunk toward CONFIDENCE_PRIOR with a weight of
# MIN_TRADES_FOR_LEARNING trades, over trades closed in the last N days (so a
# losing-streak pause expires instead of locking a strategy out forever).
CONFIDENCE_PRIOR         = 0.55
CONFIDENCE_LOOKBACK_DAYS = float(os.getenv("CONFIDENCE_LOOKBACK_DAYS", "30"))
KELLY_FRACTION          = 0.25  # fractional Kelly for position sizing

# ─── Safe self-learning (adaptive_tuner.py) — no LLM involved ────────────────
def _env_num(name: str, default, cast=float):
    return cast(os.getenv(name, str(default)))

LEARNING_ENABLED               = os.getenv("LEARNING_ENABLED", "true").lower() == "true"
LEARNING_INTERVAL_HOURS        = _env_num("LEARNING_INTERVAL_HOURS", 24)
LEARNING_MIN_NEW_TRADES        = _env_num("LEARNING_MIN_NEW_TRADES", 3, int)   # new closed trades, or…
LEARNING_MIN_NEW_DAYS          = _env_num("LEARNING_MIN_NEW_DAYS", 1)          # …days of new data
LEARNING_PROPOSAL_DAYS         = _env_num("LEARNING_PROPOSAL_DAYS", 180, int)  # window used to propose
LEARNING_VALIDATION_DAYS       = _env_num("LEARNING_VALIDATION_DAYS", 180, int)  # later, unseen window
LEARNING_WARMUP_DAYS           = _env_num("LEARNING_WARMUP_DAYS", 300, int)    # indicator warm-up
LEARNING_MIN_VALIDATION_TRADES = _env_num("LEARNING_MIN_VALIDATION_TRADES", 10, int)
LEARNING_MIN_PF_IMPROVEMENT    = _env_num("LEARNING_MIN_PF_IMPROVEMENT", 0.05)  # +5% PF out of sample
LEARNING_MAX_DD_WORSENING      = _env_num("LEARNING_MAX_DD_WORSENING", 0.02)   # max +2 pts drawdown
LEARNING_PF_CAP                = 10.0   # PF above this counts as 10 (few / no losses)
LEARNING_MAX_CHANGES_PER_DAY   = _env_num("LEARNING_MAX_CHANGES_PER_DAY", 1, int)  # per strategy
LEARNING_ROLLBACK_WINDOW_HOURS = _env_num("LEARNING_ROLLBACK_WINDOW_HOURS", 72)
LEARNING_ROLLBACK_TOLERANCE_PCT = _env_num("LEARNING_ROLLBACK_TOLERANCE_PCT", 0.02)  # of capital
BASELINE_ML_CONFIDENCE         = 0.55   # frozen baselines use a fixed ML confidence

# ─── Free market data (market_data.py) — no API key needed ────────────────────
# Binance USD-M futures public data: top-trader and all-account long/short
# ratios, taker buy/sell volume, open interest (only the last 30 days exist, so
# they are collected every hour from now on) and funding rates (years of
# history); plus the alternative.me crypto Fear & Greed index (since 2018).
BINANCE_FUTURES_BASE       = os.getenv("BINANCE_FUTURES_BASE", "https://fapi.binance.com")
FEAR_GREED_URL             = os.getenv("FEAR_GREED_URL", "https://api.alternative.me/fng/")
MARKET_DATA_INTERVAL_MIN   = _env_num("MARKET_DATA_INTERVAL_MIN", 60)
MARKET_FUNDING_BACKFILL_DAYS = _env_num("MARKET_FUNDING_BACKFILL_DAYS", 1600, int)
# Top traders beyond Binance (all public, no key): OKX top-trader ratios, the
# on-chain positions of Hyperliquid's most profitable wallets, and the CFTC
# Commitments of Traders report for CME Bitcoin / Ether futures.
OKX_BASE                   = os.getenv("OKX_BASE", "https://www.okx.com")
HL_INFO_URL                = os.getenv("HL_INFO_URL", "https://api.hyperliquid.xyz/info")
HL_LEADERBOARD_URL         = os.getenv("HL_LEADERBOARD_URL", "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard")
HL_TOP_WALLETS             = _env_num("HL_TOP_WALLETS", 150, int)          # wallets followed
HL_MIN_ACCOUNT_USD         = _env_num("HL_MIN_ACCOUNT_USD", 100_000)
HL_MAX_TURNOVER            = _env_num("HL_MAX_TURNOVER", 300)    # month volume / account: market makers above
HL_WALLET_REFRESH_HOURS    = _env_num("HL_WALLET_REFRESH_HOURS", 24)
CFTC_URL                   = os.getenv("CFTC_URL", "https://publicreporting.cftc.gov/resource/gpe5-46if.json")

# ─── Strategy evaluator (strategy_evaluator.py): viability per strategy ──────
EVAL_ENABLED            = os.getenv("EVAL_ENABLED", "true").lower() == "true"
EVAL_INTERVAL_HOURS     = _env_num("EVAL_INTERVAL_HOURS", 24)
EVAL_WINDOWS            = _env_num("EVAL_WINDOWS", 4, int)         # walk-forward windows…
EVAL_WINDOW_DAYS        = _env_num("EVAL_WINDOW_DAYS", 180, int)   # …of this many days
EVAL_WARMUP_DAYS        = _env_num("EVAL_WARMUP_DAYS", 300, int)
# VIABLE: enough trades, profit factor, profitable in most windows, bounded drawdown
VIABLE_MIN_TRADES       = _env_num("VIABLE_MIN_TRADES", 30, int)
VIABLE_MIN_PF           = _env_num("VIABLE_MIN_PF", 1.2)
VIABLE_MIN_WINDOWS      = _env_num("VIABLE_MIN_WINDOWS", 3, int)
VIABLE_MAX_DRAWDOWN     = _env_num("VIABLE_MAX_DRAWDOWN", 0.15)
# CONDICIONAL: works in some market regimes only
COND_MIN_REGIME_TRADES  = _env_num("COND_MIN_REGIME_TRADES", 10, int)
COND_MIN_REGIME_PF      = _env_num("COND_MIN_REGIME_PF", 1.3)
# DESCARTADA (value 0, never re-evaluated): strong, consistent evidence only
DISCARD_MIN_TRADES      = _env_num("DISCARD_MIN_TRADES", 40, int)
DISCARD_MAX_PF          = _env_num("DISCARD_MAX_PF", 0.9)
DISCARD_MAX_WINDOWS     = _env_num("DISCARD_MAX_WINDOWS", 1, int)   # profitable windows
DISCARD_LIVE_MIN_TRADES = _env_num("DISCARD_LIVE_MIN_TRADES", 20, int)
DISCARD_LIVE_MAX_PF     = _env_num("DISCARD_LIVE_MAX_PF", 0.7)
SIDE_BLOCK_MIN_TRADES   = 10     # a side (long/short) with >= N trades…
SIDE_BLOCK_MAX_PF       = 0.9    # …and PF below this is blocked

# Claude API for journal generation (optional — enhances reflection quality)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ─── Scheduling ────────────────────────────────────────────────────────────────
STRATEGY_CHECK_INTERVAL_SEC = 60    # check for new signals every 60s
POSITION_CHECK_INTERVAL_SEC = 20    # check SL/TP every 20s
LEARNING_UPDATE_INTERVAL_SEC = 180  # run learning update every 3 minutes

# ─── Database ──────────────────────────────────────────────────────────────────
# DATA_DIR holds the SQLite DB (and its -wal/-shm files) plus the log. In Docker
# mount a whole folder here: mounting only the .db file leaves the WAL inside
# the container, where it is lost when the container is recreated.
DATA_DIR = os.getenv("DATA_DIR") or os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_DIR, "trading_bot.db")

# ─── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_HOST      = "0.0.0.0"
DASHBOARD_PORT      = 8050
DASHBOARD_UPDATE_MS = 10000   # refresh every 10 seconds

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = os.path.join(DATA_DIR, "trading_bot.log")

# ─── Data Management ────────────────────────────────────────────────────────────
# Set to "true" to clear all trading data on startup and start fresh
# This is useful when you want to reset the bot and not show old backtest data
RESET_ON_STARTUP = os.getenv("RESET_ON_STARTUP", "false").lower() == "true"

# Set to "true" to include backtest data in dashboards (for debugging)
# Default is false - only live trading data is shown
SHOW_BACKTEST_DATA = os.getenv("SHOW_BACKTEST_DATA", "false").lower() == "true"

# ─── Strategy Parameters (defaults; self-learning engine may override) ──────────
STRATEGY_PARAMS = {
    "EMA5_Momentum": {
        # Source  : Quantified Strategies – best EMA period for Bitcoin (~145% CAGR)
        # Entry LONG  : close crosses above 5-day EMA
        # Entry SHORT : close crosses below 5-day EMA
        "ema_period":      5,
        "atr_sl_mult":     1.5,    # SL = 1.5 × ATR below entry
        "atr_tp_mult":     3.5,    # TP = 3.5 × ATR above entry
        "candle_interval": "1d",
    },
    "DualMA_Crossover": {
        # Source  : Quantified Strategies – 100/250 SMA crossover (~115% CAGR)
        # Entry LONG  : SMA-100 crosses above SMA-250 (golden cross)
        # Entry SHORT : SMA-100 crosses below SMA-250 (death  cross)
        # Requires BACKTEST_DAYS >= 300 for SMA-250 warm-up
        "fast_period":     100,
        "slow_period":     250,
        "atr_sl_mult":     2.0,
        "atr_tp_mult":     5.0,
        "candle_interval": "1d",
    },
    "Regime_RiskOnOff": {
        # Source  : Menthor Q – binary risk-on/risk-off model (~100-200% cumulative/yr)
        # Proxy   : EMA-200 + MACD histogram + RSI all must agree (on-chain metrics
        #           not available via Binance REST API)
        # Entry LONG  : all three conditions bullish (regime switches to RISK-ON)
        # Entry SHORT : all three conditions bearish (regime switches to RISK-OFF)
        "ema_trend":       200,
        "rsi_bull_min":    50,
        "rsi_bear_max":    50,
        "atr_sl_mult":     2.0,
        "atr_tp_mult":     4.5,
        "candle_interval": "4h",
    },
    "PriceMomentum_25": {
        # Source  : Quantified Strategies – 25-day close-to-close momentum (~115% CAGR)
        # Entry LONG  : today's close > close 25 days ago
        # Entry SHORT : today's close < close 25 days ago
        "lookback":        25,
        "atr_sl_mult":     1.5,
        "atr_tp_mult":     4.0,
        "candle_interval": "1d",
    },
    "Residual_MeanRev": {
        # Source  : Medium – BTC-neutral residual mean reversion (Sharpe ~2.3 post-2021)
        # Proxy   : rolling OLS regression on log-price; trade deviations from trend
        #           (original uses altcoin-vs-BTC beta stripping; here we strip BTC's
        #            own trend since we only trade BTCUSDT)
        # Entry LONG  : z-score of residual < -1.5 (below trend, oversold)
        # Entry SHORT : z-score of residual > +1.5 (above trend, overbought)
        "reg_window":      60,
        "zscore_window":   30,
        "entry_threshold": 1.5,
        "atr_sl_mult":     1.8,
        "atr_tp_mult":     3.5,
        "candle_interval": "4h",
    },
    "Donchian_Breakout": {
        # Source  : Quantified Strategies – Donchian breakout on BTC/USD (back to 2015)
        # INVERSE ADX: enter when ADX < threshold (market is calm / consolidating)
        # 15-day lookback offers best risk/reward per research
        # Entry LONG  : close > previous 15-day Donchian upper  AND  ADX < 25
        # Entry SHORT : close < previous 15-day Donchian lower  AND  ADX < 25
        "dc_period":       15,
        "adx_calm_max":    25,
        "atr_sl_mult":     1.5,
        "atr_tp_mult":     3.5,
        "candle_interval": "1d",
    },
    "Blended_MomentumMR": {
        # Source  : Medium – 50/50 momentum + mean-reversion portfolio (best risk-adj)
        # Momentum: 25-period close-to-close (pre-2021 dominant)
        # MR      : RSI + Bollinger Bands (post-2021 dominant)
        # Blend for regime-robust performance across all market cycles
        "momentum_period":  25,
        "rsi_oversold":     38,
        "rsi_overbought":   62,
        "bb_period":        20,
        "bb_std":           2.0,
        "atr_sl_mult":      1.8,
        "atr_tp_mult":      3.8,
        "candle_interval":  "4h",
    },
    "BTC_MomentumBreakout": {
        # Source  : Vault – BTC momentum breakout (backtested 2018-2026, +42% CAGR)
        # Entry   : close > 200d EMA  AND  close > 20d close-high  AND  volume > 1.2× vol MA
        # Exit    : close < 10d low  OR  close < 50d SMA  OR  -8% hard stop
        # TP/SL   : ATR-based take-profit (3× ATR), dual SL (ATR-mult + -8% hard cap)
        # Alpha   : catches sustained BTC breakouts in bull regimes; avoids chop
        "sma_trend":          200,
        "breakout_lookback":   20,
        "volume_ma_period":    20,
        "volume_mult":        1.2,
        "exit_low_period":    10,
        "sma_exit_period":    50,
        "hard_stop_pct":     0.08,
        "atr_tp_mult":       3.0,
        "atr_sl_mult":       1.5,
        "candle_interval":  "1d",
    },

    # ── Candidate catalog (strategies/__init__.py CANDIDATE_STRATEGIES) ──────
    # Rated by strategy_evaluator.py; they only trade in the learning book once
    # rated VIABLE (or CONDITIONAL, in a market regime where they work).
    # The first four already existed in the repo but crashed at construction
    # (no entry here); their values are the ones that were hard-coded.
    "RSI_Bollinger": {
        "rsi_period": 14, "bb_period": 20,
        "rsi_oversold": 30, "rsi_overbought": 70,
        "candle_interval": "4h",
    },
    "MACD_Momentum": {
        "trend_ema": 200, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
        "candle_interval": "1h",
    },
    "EMA_Crossover": {
        "trend_ema": 50, "atr_sl_mult": 2.0, "atr_tp_mult": 4.0,
        "candle_interval": "1h",
    },
    "Breakout": {
        "lookback": 24, "atr_period": 14, "volume_multiplier": 1.8,
        "atr_tp_mult": 2.5, "candle_interval": "4h",
    },
    "Turtle_Breakout": {
        "entry_period": 20, "atr_sl_mult": 2.0, "atr_tp_mult": 4.0,
        "candle_interval": "1d",
    },
    "Connors_RSI2": {
        "rsi_low": 10, "rsi_high": 90, "atr_sl_mult": 2.0, "atr_tp_mult": 1.5,
        "candle_interval": "1d",
    },
    "Bollinger_Squeeze": {
        "bb_period": 20, "bb_std": 2.0, "squeeze_lookback": 120, "squeeze_quantile": 0.15,
        "atr_sl_mult": 1.5, "atr_tp_mult": 3.0, "candle_interval": "4h",
    },
    "Supertrend": {
        "atr_period": 10, "multiplier": 3.0, "atr_tp_mult": 3.0,
        "candle_interval": "4h",
    },
    "Golden_Cross_50_200": {
        "fast_period": 50, "slow_period": 200, "atr_sl_mult": 2.5, "atr_tp_mult": 6.0,
        "candle_interval": "1d",
    },
    # Modern catalog (strategies/modern_catalog.py) — free market data + books
    "FearGreed_Contrarian": {
        "fear_max": 20, "greed_min": 80, "atr_sl_mult": 2.0, "atr_tp_mult": 3.0,
        "candle_interval": "1d",
    },
    "Funding_Contrarian": {
        "funding_high": 0.0005, "funding_low": -0.0001, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
        "candle_interval": "4h",
    },
    "TopTraders_Follow": {
        "z_entry": 1.5, "lookback": 72, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
        "candle_interval": "1h",
    },
    "Crowd_vs_TopTraders": {
        "z_entry": 1.5, "lookback": 72, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
        "candle_interval": "1h",
    },
    "Carver_EWMAC": {
        "fast_period": 16, "threshold": 10, "atr_sl_mult": 2.0, "atr_tp_mult": 4.0,
        "candle_interval": "1d",
    },
    "SmartMoney_Consensus": {
        "z_entry": 1.0, "lookback": 72, "hl_net_min": 0.3, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
        "candle_interval": "1h",
    },
    "HL_TopWallets_Follow": {
        "net_entry": 0.4, "swing": 0.25, "swing_hours": 6, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0,
        "candle_interval": "1h",
    },
    "COT_Institucional": {
        "weeks": 26, "index_high": 80, "index_low": 20, "atr_sl_mult": 2.0, "atr_tp_mult": 4.0,
        "candle_interval": "1d",
    },
}


# ─── Configuration Validation ───────────────────────────────────────────────────
def validate_config() -> list:
    """
    Validate configuration settings.
    
    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []
    
    # Validate capital
    if INITIAL_CAPITAL < 100:
        errors.append(f"INITIAL_CAPITAL must be >= $100, got ${INITIAL_CAPITAL}")
    
    # Validate risk parameters
    if DEFAULT_STOP_LOSS_PCT <= 0 or DEFAULT_STOP_LOSS_PCT >= 1:
        errors.append(f"DEFAULT_STOP_LOSS_PCT must be between 0 and 1, got {DEFAULT_STOP_LOSS_PCT}")
    
    if DEFAULT_TAKE_PROFIT_PCT <= 0 or DEFAULT_TAKE_PROFIT_PCT >= 1:
        errors.append(f"DEFAULT_TAKE_PROFIT_PCT must be between 0 and 1, got {DEFAULT_TAKE_PROFIT_PCT}")
    
    if DEFAULT_TAKE_PROFIT_PCT <= DEFAULT_STOP_LOSS_PCT:
        errors.append(f"DEFAULT_TAKE_PROFIT_PCT ({DEFAULT_TAKE_PROFIT_PCT}) must be > DEFAULT_STOP_LOSS_PCT ({DEFAULT_STOP_LOSS_PCT})")
    
    # Validate position sizing
    if MAX_POSITION_PCT <= 0 or MAX_POSITION_PCT > 1:
        errors.append(f"MAX_POSITION_PCT must be between 0 and 1, got {MAX_POSITION_PCT}")
    
    # Validate backtest parameters
    if BACKTEST_DAYS < 100:
        errors.append(f"BACKTEST_DAYS should be >= 100 for meaningful results, got {BACKTEST_DAYS}")
    
    if MIN_CAGR_THRESHOLD < 0:
        errors.append(f"MIN_CAGR_THRESHOLD must be >= 0, got {MIN_CAGR_THRESHOLD}")
    
    # Validate ML parameters
    if MIN_TRADES_FOR_LEARNING < 5:
        errors.append(f"MIN_TRADES_FOR_LEARNING should be >= 5, got {MIN_TRADES_FOR_LEARNING}")
    
    if not 0 <= CONFIDENCE_THRESHOLD <= 1:
        errors.append(f"CONFIDENCE_THRESHOLD must be between 0 and 1, got {CONFIDENCE_THRESHOLD}")
    
    # Validate API credentials for live trading
    if not PAPER_TRADING:
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            errors.append("BINANCE_API_KEY and BINANCE_API_SECRET required for live trading")
    
    # Validate symbol format
    if not SYMBOL.endswith(("USDT", "BUSD", "USD")):
        errors.append(f"SYMBOL should end with USDT/BUSD/USD, got {SYMBOL}")
    
    return errors


def get_config_summary() -> dict:
    """Get a summary of the current configuration."""
    return {
        "symbol": SYMBOL,
        "capital": f"${INITIAL_CAPITAL:,.2f}",
        "mode": "PAPER" if PAPER_TRADING else "LIVE",
        "stop_loss": f"{DEFAULT_STOP_LOSS_PCT*100:.1f}%",
        "take_profit": f"{DEFAULT_TAKE_PROFIT_PCT*100:.1f}%",
        "max_position": f"{MAX_POSITION_PCT*100:.0f}%",
        "backtest_days": BACKTEST_DAYS,
        "strategies": len(STRATEGY_PARAMS),
    }


# Run validation on import
_config_errors = validate_config()
if _config_errors:
    import logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("config")
    for error in _config_errors:
        logger.warning(f"Config validation: {error}")
