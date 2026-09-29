from .base_strategy import BaseStrategy, Signal, SignalType
from .ema5_momentum import EMA5MomentumStrategy
from .dual_ma_crossover import DualMACrossoverStrategy
from .regime_riskoff import RegimeRiskOffStrategy
from .price_momentum_25 import PriceMomentum25Strategy
from .residual_mean_reversion import ResidualMeanReversionStrategy
from .donchian_breakout import DonchianBreakoutStrategy
from .blended_momentum_mr import BlendedMomentumMRStrategy
from .btc_momentum_breakout import BTCMomentumBreakoutStrategy
from .rsi_bollinger import RSIBollingerStrategy
from .macd_momentum import MACDMomentumStrategy
from .ema_crossover import EMACrossoverStrategy
from .breakout import BreakoutStrategy
from .classic_catalog import (TurtleBreakoutStrategy, ConnorsRSI2Strategy, BollingerSqueezeStrategy,
                              SupertrendStrategy, GoldenCross50200Strategy)
from .modern_catalog import (FearGreedContrarianStrategy, FundingContrarianStrategy,
                             TopTradersFollowStrategy, CrowdVsTopTradersStrategy, CarverEWMACStrategy)
from .smart_money_catalog import (SmartMoneyConsensusStrategy, HLTopWalletsFollowStrategy,
                                  COTInstitutionalStrategy)

ALL_STRATEGIES = [
    EMA5MomentumStrategy,           # 1 – Short-window EMA momentum     (~145% CAGR)
    DualMACrossoverStrategy,        # 2 – 100/250 SMA dual crossover     (~115% CAGR)
    RegimeRiskOffStrategy,          # 3 – Risk-On/Off regime model       (variable, high)
    PriceMomentum25Strategy,        # 4 – 25-day close-to-close momentum (~115% CAGR)
    ResidualMeanReversionStrategy,  # 5 – Residual mean reversion        (Sharpe ~2.3)
    DonchianBreakoutStrategy,        # 6 – Donchian breakout + inverse ADX (competitive)
    BlendedMomentumMRStrategy,      # 7 – 50/50 momentum + MR blend      (best risk-adj)
    BTCMomentumBreakoutStrategy,    # 8 – BTC momentum breakout          (+42% CAGR vault)
]

# Candidate catalog: evaluated by strategy_evaluator.py (walk-forward + market
# regime); they trade in the learning book only once rated VIABLE/CONDITIONAL.
# New entries are added by hand, with tests and a SOURCE — never downloaded code.
CANDIDATE_STRATEGIES = [
    RSIBollingerStrategy,           # repo, previously unregistered
    MACDMomentumStrategy,           # repo, previously unregistered
    EMACrossoverStrategy,           # repo, previously unregistered
    BreakoutStrategy,               # repo, previously unregistered
    TurtleBreakoutStrategy,         # Turtle System 1
    ConnorsRSI2Strategy,            # Connors RSI(2)
    BollingerSqueezeStrategy,       # Bollinger Squeeze
    SupertrendStrategy,             # Supertrend 10 x 3
    GoldenCross50200Strategy,       # 50/200 golden cross
    # Modern: free positioning / funding / sentiment data (market_data.py) + books
    FearGreedContrarianStrategy,    # Fear & Greed extremes, contrarian
    FundingContrarianStrategy,      # crowded funding, contrarian
    TopTradersFollowStrategy,       # follow Binance top traders' position build-ups
    CrowdVsTopTradersStrategy,      # fade the crowd when top traders disagree
    CarverEWMACStrategy,            # Carver, Systematic Trading: EWMAC 16/64
    # Top traders beyond Binance: OKX, Hyperliquid on-chain wallets, CME (CFTC COT)
    SmartMoneyConsensusStrategy,    # Binance + OKX + Hyperliquid top traders agree
    HLTopWalletsFollowStrategy,     # follow Hyperliquid's most profitable wallets
    COTInstitutionalStrategy,       # Larry Williams' COT index on CME asset managers
]
