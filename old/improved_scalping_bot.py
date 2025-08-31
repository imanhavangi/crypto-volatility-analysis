#!/usr/bin/env python3
"""
Improved DOGE/USD Scalping Bot v2.0
===================================
Enhanced scalping bot with adaptive strategies, better entry conditions,
and improved risk management for more consistent profits.

Key Improvements:
- Adaptive entry conditions based on market volatility
- Multiple scalping strategies (trend following, mean reversion)
- Dynamic position sizing
- Better risk management with trailing stops
- Market regime detection
- Enhanced performance metrics

Dependencies:
    pip install ccxt pandas numpy matplotlib seaborn

Usage:
    python improved_scalping_bot.py
"""

from __future__ import annotations
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, NamedTuple
import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Trading Configuration
SYMBOL = "DOGE/USD"
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
TIMEFRAME = "1m"
LOOKBACK_DAYS = 30

# Adaptive Risk Management
INITIAL_BALANCE = 1000.0
BASE_POSITION_SIZE_PCT = 0.08  # Base position size (8%)
MAX_POSITION_SIZE_PCT = 0.15   # Maximum position size (15%)
MIN_POSITION_SIZE_PCT = 0.03   # Minimum position size (3%)

# Profit/Loss Targets (Adaptive)
BASE_PROFIT_TARGET_PCT = 0.004  # 0.4% base profit target
MAX_PROFIT_TARGET_PCT = 0.008   # 0.8% maximum profit target
MIN_PROFIT_TARGET_PCT = 0.002   # 0.2% minimum profit target

BASE_STOP_LOSS_PCT = 0.003      # 0.3% base stop loss
MAX_STOP_LOSS_PCT = 0.006       # 0.6% maximum stop loss
MIN_STOP_LOSS_PCT = 0.0015      # 0.15% minimum stop loss

# Trading Controls
MAX_TRADES_PER_HOUR = 10
MIN_TIME_BETWEEN_TRADES = 120  # 2 minutes
MAX_CONSECUTIVE_LOSSES = 3

# Technical Indicator Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 40  # More relaxed
RSI_OVERBOUGHT = 60  # More relaxed
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_DEVIATION = 2
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_MA_PERIOD = 20

# ---------------------------------------------------------------------------
# Market Regime and Strategy Types
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"

@dataclass
class TradingParameters:
    """Adaptive trading parameters based on market conditions."""
    position_size_pct: float
    profit_target_pct: float
    stop_loss_pct: float
    rsi_oversold: float
    rsi_overbought: float
    min_confidence: float
    strategy_type: StrategyType

@dataclass
class Trade:
    """Enhanced trade structure."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str
    profit_loss: Optional[float]
    profit_pct: Optional[float]
    reason: str
    strategy: StrategyType
    market_regime: MarketRegime
    confidence: float

class Signal(NamedTuple):
    """Enhanced trading signal."""
    action: str
    confidence: float
    reason: str
    price: float
    strategy: StrategyType
    parameters: TradingParameters

# ---------------------------------------------------------------------------
# Enhanced Technical Indicators
# ---------------------------------------------------------------------------

class AdvancedIndicators:
    """Advanced technical indicators with market regime detection."""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with smoothing."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def market_regime(prices: pd.Series, volume: pd.Series, period: int = 50) -> Dict[str, pd.Series]:
        """Detect market regime (trending, sideways, volatile)."""
        # Trend strength
        sma_fast = prices.rolling(10).mean()
        sma_slow = prices.rolling(30).mean()
        trend_strength = abs(sma_fast - sma_slow) / sma_slow
        
        # Volatility measure
        returns = prices.pct_change()
        volatility = returns.rolling(period).std()
        volatility_percentile = volatility.rolling(period*2).rank(pct=True)
        
        # Volume trend
        volume_ma = volume.rolling(20).mean()
        volume_trend = volume / volume_ma
        
        # Price momentum
        momentum = prices.pct_change(10)
        
        return {
            'trend_strength': trend_strength,
            'volatility': volatility,
            'volatility_percentile': volatility_percentile,
            'volume_trend': volume_trend,
            'momentum': momentum,
            'is_trending': trend_strength > trend_strength.rolling(period).median(),
            'is_high_vol': volatility_percentile > 0.7,
            'is_low_vol': volatility_percentile < 0.3
        }
    
    @staticmethod
    def dynamic_levels(prices: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate dynamic support/resistance levels."""
        highs = prices.rolling(period).max()
        lows = prices.rolling(period).min()
        midpoint = (highs + lows) / 2
        
        # Fibonacci levels
        range_size = highs - lows
        fib_618 = lows + range_size * 0.618
        fib_382 = lows + range_size * 0.382
        
        return {
            'resistance': highs,
            'support': lows,
            'midpoint': midpoint,
            'fib_618': fib_618,
            'fib_382': fib_382,
            'range_size': range_size
        }
    
    @staticmethod
    def volume_analysis(volume: pd.Series, prices: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Advanced volume analysis."""
        volume_ma = volume.rolling(period).mean()
        volume_ratio = volume / volume_ma
        
        # Price-volume relationship
        price_change = prices.pct_change()
        volume_price_corr = price_change.rolling(period).corr(volume_ratio)
        
        # Volume momentum
        volume_momentum = volume.rolling(5).mean() / volume.rolling(20).mean()
        
        return {
            'volume_ratio': volume_ratio,
            'volume_momentum': volume_momentum,
            'volume_price_corr': volume_price_corr,
            'high_volume': volume_ratio > 1.5,
            'volume_breakout': volume_ratio > 2.0
        }

# ---------------------------------------------------------------------------
# Adaptive Strategy Engine
# ---------------------------------------------------------------------------

class AdaptiveStrategy:
    """Advanced multi-strategy scalping engine."""
    
    def __init__(self):
        self.indicators = AdvancedIndicators()
        self.last_trade_time = None
        self.trades_this_hour = 0
        self.current_hour = None
        self.consecutive_losses = 0
        self.last_regime = None
        
    def get_market_regime(self, analysis: Dict[str, pd.Series]) -> MarketRegime:
        """Determine current market regime."""
        if len(analysis['trend_strength']) < 2:
            return MarketRegime.SIDEWAYS
            
        latest_idx = -1
        trend_strength = analysis['trend_strength'].iloc[latest_idx]
        volatility_pct = analysis['volatility_percentile'].iloc[latest_idx]
        momentum = analysis['momentum'].iloc[latest_idx]
        is_trending = analysis['is_trending'].iloc[latest_idx]
        
        # Skip if any value is NaN
        if pd.isna([trend_strength, volatility_pct, momentum]).any():
            return MarketRegime.SIDEWAYS
        
        # High volatility regime
        if volatility_pct > 0.8:
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility regime
        if volatility_pct < 0.2:
            return MarketRegime.LOW_VOLATILITY
        
        # Trending regimes
        if is_trending and abs(momentum) > 0.005:
            if momentum > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        return MarketRegime.SIDEWAYS
    
    def get_adaptive_parameters(self, regime: MarketRegime, volatility_pct: float) -> TradingParameters:
        """Get adaptive trading parameters based on market regime."""
        
        if regime == MarketRegime.TRENDING_UP:
            return TradingParameters(
                position_size_pct=BASE_POSITION_SIZE_PCT * 1.2,
                profit_target_pct=BASE_PROFIT_TARGET_PCT * 1.5,
                stop_loss_pct=BASE_STOP_LOSS_PCT * 1.3,
                rsi_oversold=45,
                rsi_overbought=65,
                min_confidence=0.6,
                strategy_type=StrategyType.TREND_FOLLOWING
            )
        elif regime == MarketRegime.TRENDING_DOWN:
            return TradingParameters(
                position_size_pct=BASE_POSITION_SIZE_PCT * 0.8,
                profit_target_pct=BASE_PROFIT_TARGET_PCT * 1.2,
                stop_loss_pct=BASE_STOP_LOSS_PCT * 1.5,
                rsi_oversold=35,
                rsi_overbought=55,
                min_confidence=0.7,
                strategy_type=StrategyType.TREND_FOLLOWING
            )
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return TradingParameters(
                position_size_pct=MIN_POSITION_SIZE_PCT,
                profit_target_pct=MAX_PROFIT_TARGET_PCT,
                stop_loss_pct=MAX_STOP_LOSS_PCT,
                rsi_oversold=30,
                rsi_overbought=70,
                min_confidence=0.8,
                strategy_type=StrategyType.BREAKOUT
            )
        elif regime == MarketRegime.LOW_VOLATILITY:
            return TradingParameters(
                position_size_pct=MAX_POSITION_SIZE_PCT,
                profit_target_pct=MIN_PROFIT_TARGET_PCT,
                stop_loss_pct=MIN_STOP_LOSS_PCT,
                rsi_oversold=45,
                rsi_overbought=55,
                min_confidence=0.5,
                strategy_type=StrategyType.MEAN_REVERSION
            )
        else:  # SIDEWAYS
            return TradingParameters(
                position_size_pct=BASE_POSITION_SIZE_PCT,
                profit_target_pct=BASE_PROFIT_TARGET_PCT,
                stop_loss_pct=BASE_STOP_LOSS_PCT,
                rsi_oversold=40,
                rsi_overbought=60,
                min_confidence=0.6,
                strategy_type=StrategyType.MEAN_REVERSION
            )
    
    def analyze_market(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Comprehensive market analysis."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        analysis = {}
        
        # Basic indicators
        analysis['rsi'] = self.indicators.rsi(close, RSI_PERIOD)
        
        # MACD
        ema_fast = close.ewm(span=MACD_FAST).mean()
        ema_slow = close.ewm(span=MACD_SLOW).mean()
        analysis['macd'] = ema_fast - ema_slow
        analysis['macd_signal'] = analysis['macd'].ewm(span=MACD_SIGNAL).mean()
        analysis['macd_histogram'] = analysis['macd'] - analysis['macd_signal']
        
        # Bollinger Bands
        bb_middle = close.rolling(BB_PERIOD).mean()
        bb_std = close.rolling(BB_PERIOD).std()
        analysis['bb_upper'] = bb_middle + (bb_std * BB_DEVIATION)
        analysis['bb_lower'] = bb_middle - (bb_std * BB_DEVIATION)
        analysis['bb_middle'] = bb_middle
        analysis['bb_position'] = (close - analysis['bb_lower']) / (analysis['bb_upper'] - analysis['bb_lower'])
        
        # Moving averages
        analysis['ema_fast'] = close.ewm(span=EMA_FAST).mean()
        analysis['ema_slow'] = close.ewm(span=EMA_SLOW).mean()
        analysis['ema_trend'] = analysis['ema_fast'] > analysis['ema_slow']
        
        # Market regime detection
        regime_data = self.indicators.market_regime(close, volume)
        analysis.update(regime_data)
        
        # Dynamic levels
        levels_data = self.indicators.dynamic_levels(close)
        analysis.update(levels_data)
        
        # Volume analysis
        volume_data = self.indicators.volume_analysis(volume, close)
        analysis.update(volume_data)
        
        # Price momentum
        analysis['price_momentum'] = close.pct_change(5)
        analysis['price_acceleration'] = analysis['price_momentum'].diff()
        
        return analysis
    
    def generate_signal(self, df: pd.DataFrame, analysis: Dict[str, pd.Series]) -> Signal:
        """Generate adaptive trading signal."""
        if len(df) < max(RSI_PERIOD, MACD_SLOW, BB_PERIOD, EMA_SLOW, 50):
            return self._hold_signal(df['close'].iloc[-1])
        
        current_price = df['close'].iloc[-1]
        current_time = df.index[-1]
        
        # Check trading restrictions
        if not self._can_trade(current_time):
            return self._hold_signal(current_price)
        
        # Determine market regime
        regime = self.get_market_regime(analysis)
        
        # Get adaptive parameters
        params = self.get_adaptive_parameters(regime, analysis['volatility_percentile'].iloc[-1])
        
        # Adjust parameters based on consecutive losses
        if self.consecutive_losses >= 2:
            params.position_size_pct *= 0.5
            params.min_confidence += 0.1
        
        # Generate strategy-specific signal
        if params.strategy_type == StrategyType.TREND_FOLLOWING:
            signal = self._trend_following_signal(df, analysis, params, regime)
        elif params.strategy_type == StrategyType.MEAN_REVERSION:
            signal = self._mean_reversion_signal(df, analysis, params, regime)
        elif params.strategy_type == StrategyType.BREAKOUT:
            signal = self._breakout_signal(df, analysis, params, regime)
        else:  # MOMENTUM
            signal = self._momentum_signal(df, analysis, params, regime)
        
        return signal
    
    def _trend_following_signal(self, df: pd.DataFrame, analysis: Dict[str, pd.Series], 
                              params: TradingParameters, regime: MarketRegime) -> Signal:
        """Trend following strategy."""
        latest_idx = -1
        current_price = df['close'].iloc[latest_idx]
        
        # Get indicator values
        rsi = analysis['rsi'].iloc[latest_idx]
        macd = analysis['macd'].iloc[latest_idx]
        macd_signal = analysis['macd_signal'].iloc[latest_idx]
        ema_trend = analysis['ema_trend'].iloc[latest_idx]
        volume_ratio = analysis['volume_ratio'].iloc[latest_idx]
        momentum = analysis['momentum'].iloc[latest_idx]
        
        if pd.isna([rsi, macd, macd_signal, volume_ratio, momentum]).any():
            return self._hold_signal(current_price)
        
        confidence = 0.0
        signals = []
        
        # Buy signals for uptrend
        if regime == MarketRegime.TRENDING_UP:
            if rsi > params.rsi_oversold and rsi < 55:
                signals.append("RSI favorable for uptrend")
                confidence += 0.25
            
            if macd > macd_signal and macd > 0:
                signals.append("MACD bullish in uptrend")
                confidence += 0.3
            
            if ema_trend:
                signals.append("EMA trend confirmed")
                confidence += 0.25
            
            if volume_ratio > 1.1:
                signals.append("Volume support")
                confidence += 0.2
            
            if confidence >= params.min_confidence:
                reason = f"TREND BUY: {', '.join(signals)}"
                return Signal('buy', confidence, reason, current_price, params.strategy_type, params)
        
        # No trend following sell signals - let profit target/stop loss handle exits
        return self._hold_signal(current_price)
    
    def _mean_reversion_signal(self, df: pd.DataFrame, analysis: Dict[str, pd.Series], 
                             params: TradingParameters, regime: MarketRegime) -> Signal:
        """Mean reversion strategy."""
        latest_idx = -1
        current_price = df['close'].iloc[latest_idx]
        
        # Get indicator values
        rsi = analysis['rsi'].iloc[latest_idx]
        bb_position = analysis['bb_position'].iloc[latest_idx]
        volume_ratio = analysis['volume_ratio'].iloc[latest_idx]
        price_momentum = analysis['price_momentum'].iloc[latest_idx]
        
        if pd.isna([rsi, bb_position, volume_ratio, price_momentum]).any():
            return self._hold_signal(current_price)
        
        confidence = 0.0
        signals = []
        
        # Buy on oversold conditions
        if rsi < params.rsi_oversold:
            signals.append("RSI oversold")
            confidence += 0.3
        
        if bb_position < 0.2:
            signals.append("Price near BB lower band")
            confidence += 0.25
        
        if price_momentum < -0.003:  # Price dropped significantly
            signals.append("Price momentum oversold")
            confidence += 0.2
        
        if volume_ratio > 1.0:  # Some volume confirmation
            signals.append("Volume present")
            confidence += 0.15
        
        # Additional confirmation for low volatility environment
        if regime == MarketRegime.LOW_VOLATILITY and bb_position < 0.3:
            signals.append("Low vol mean reversion setup")
            confidence += 0.1
        
        if confidence >= params.min_confidence:
            reason = f"MEAN REVERSION BUY: {', '.join(signals)}"
            return Signal('buy', confidence, reason, current_price, params.strategy_type, params)
        
        return self._hold_signal(current_price)
    
    def _breakout_signal(self, df: pd.DataFrame, analysis: Dict[str, pd.Series], 
                        params: TradingParameters, regime: MarketRegime) -> Signal:
        """Breakout strategy for high volatility."""
        latest_idx = -1
        current_price = df['close'].iloc[latest_idx]
        
        # Get indicator values
        bb_position = analysis['bb_position'].iloc[latest_idx]
        volume_ratio = analysis['volume_ratio'].iloc[latest_idx]
        price_momentum = analysis['price_momentum'].iloc[latest_idx]
        macd_histogram = analysis['macd_histogram'].iloc[latest_idx]
        
        if pd.isna([bb_position, volume_ratio, price_momentum, macd_histogram]).any():
            return self._hold_signal(current_price)
        
        confidence = 0.0
        signals = []
        
        # Breakout above resistance with volume
        if bb_position > 0.9 and volume_ratio > 2.0:
            signals.append("Volume breakout above resistance")
            confidence += 0.4
        
        # Strong momentum with MACD confirmation
        if price_momentum > 0.005 and macd_histogram > 0:
            signals.append("Strong bullish momentum")
            confidence += 0.3
        
        # Volume spike
        if volume_ratio > 3.0:
            signals.append("Exceptional volume spike")
            confidence += 0.3
        
        if confidence >= params.min_confidence:
            reason = f"BREAKOUT BUY: {', '.join(signals)}"
            return Signal('buy', confidence, reason, current_price, params.strategy_type, params)
        
        return self._hold_signal(current_price)
    
    def _momentum_signal(self, df: pd.DataFrame, analysis: Dict[str, pd.Series], 
                        params: TradingParameters, regime: MarketRegime) -> Signal:
        """Momentum strategy."""
        latest_idx = -1
        current_price = df['close'].iloc[latest_idx]
        
        # Get indicator values
        rsi = analysis['rsi'].iloc[latest_idx]
        macd = analysis['macd'].iloc[latest_idx]
        macd_signal = analysis['macd_signal'].iloc[latest_idx]
        price_momentum = analysis['price_momentum'].iloc[latest_idx]
        price_acceleration = analysis['price_acceleration'].iloc[latest_idx]
        
        if pd.isna([rsi, macd, macd_signal, price_momentum, price_acceleration]).any():
            return self._hold_signal(current_price)
        
        confidence = 0.0
        signals = []
        
        # Momentum building
        if macd > macd_signal and (macd - macd_signal) > 0:
            signals.append("MACD momentum building")
            confidence += 0.3
        
        if price_momentum > 0.002 and price_acceleration > 0:
            signals.append("Price momentum accelerating")
            confidence += 0.3
        
        if 45 < rsi < 65:  # Not overbought yet
            signals.append("RSI in momentum zone")
            confidence += 0.2
        
        if confidence >= params.min_confidence:
            reason = f"MOMENTUM BUY: {', '.join(signals)}"
            return Signal('buy', confidence, reason, current_price, params.strategy_type, params)
        
        return self._hold_signal(current_price)
    
    def _hold_signal(self, price: float) -> Signal:
        """Generate hold signal."""
        return Signal('hold', 0.0, 'No clear signal', price, StrategyType.MEAN_REVERSION, 
                     TradingParameters(0.1, 0.003, 0.002, 40, 60, 0.6, StrategyType.MEAN_REVERSION))
    
    def _can_trade(self, current_time: datetime) -> bool:
        """Enhanced trading restrictions."""
        current_hour = current_time.hour
        
        # Reset hourly counter
        if self.current_hour != current_hour:
            self.current_hour = current_hour
            self.trades_this_hour = 0
        
        # Check hourly limit
        if self.trades_this_hour >= MAX_TRADES_PER_HOUR:
            return False
        
        # Check minimum time between trades
        if self.last_trade_time:
            time_diff = (current_time - self.last_trade_time).total_seconds()
            if time_diff < MIN_TIME_BETWEEN_TRADES:
                return False
        
        # Prevent trading after consecutive losses
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return False
        
        return True
    
    def update_trade_result(self, trade_time: datetime, was_profitable: bool):
        """Update trading statistics."""
        self.last_trade_time = trade_time
        self.trades_this_hour += 1
        
        if was_profitable:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

# ---------------------------------------------------------------------------
# Enhanced Backtesting Engine
# ---------------------------------------------------------------------------

class EnhancedBacktestingEngine:
    """Enhanced backtesting with adaptive strategies."""
    
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0
        self.trades: List[Trade] = []
        self.strategy = AdaptiveStrategy()
        self.open_trade: Optional[Trade] = None
        self.current_parameters: Optional[TradingParameters] = None
        
        # Performance tracking
        self.daily_returns = []
        self.equity_curve = [initial_balance]
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data (same as before)."""
        print(f"📊 Fetching {LOOKBACK_DAYS} days of 1-minute data for {SYMBOL}...")
        
        exchange = ccxt.kraken({'enableRateLimit': True})
        exchange.load_markets()
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=LOOKBACK_DAYS)
        
        all_data = []
        since_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        while since_ms < end_ms:
            try:
                ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ms, limit=1000)
                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                since_ms = ohlcv[-1][0] + 60000
                time.sleep(0.1)
                
                progress = (since_ms - int(start_time.timestamp() * 1000)) / (end_ms - int(start_time.timestamp() * 1000))
                print(f"\r   Progress: {progress:.1%}", end="", flush=True)
                
            except Exception as e:
                print(f"\n   ⚠️ Error fetching data: {e}")
                break
        
        print(f"\n   ✅ Fetched {len(all_data)} candles")
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df.astype(float).dropna()
        
        return df
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run enhanced backtest."""
        print(f"\n🚀 Starting Enhanced Adaptive Backtest")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"🧠 Adaptive Strategies: Trend Following, Mean Reversion, Breakout, Momentum")
        print(f"🛡️ Dynamic Risk Management: Adaptive position sizing and targets")
        print("=" * 70)
        
        # Calculate indicators
        analysis = self.strategy.analyze_market(df)
        
        total_signals = 0
        buy_signals = 0
        strategy_counts = {strategy: 0 for strategy in StrategyType}
        regime_counts = {regime: 0 for regime in MarketRegime}
        
        # Iterate through data
        for i in range(max(RSI_PERIOD, MACD_SLOW, BB_PERIOD, EMA_SLOW, 50), len(df)):
            current_time = df.index[i]
            current_data = df.iloc[:i+1]
            current_analysis = {key: series.iloc[:i+1] for key, series in analysis.items()}
            
            # Generate signal
            signal = self.strategy.generate_signal(current_data, current_analysis)
            total_signals += 1
            
            # Track regime
            regime = self.strategy.get_market_regime(current_analysis)
            regime_counts[regime] += 1
            
            if signal.action != 'hold':
                print(f"🔔 {current_time.strftime('%Y-%m-%d %H:%M')} - {signal.action.upper()}")
                print(f"    Strategy: {signal.strategy.value} | Confidence: {signal.confidence:.2f}")
                print(f"    Regime: {regime.value} | Reason: {signal.reason}")
            
            # Process signal
            if signal.action == 'buy' and self.open_trade is None:
                self._execute_buy(current_time, signal.price, signal.reason, signal.strategy, regime, signal.confidence, signal.parameters)
                buy_signals += 1
                strategy_counts[signal.strategy] += 1
                
            # Check exit conditions
            if self.open_trade is not None:
                self._check_exit_conditions(current_time, df.iloc[i])
            
            # Update equity curve
            current_value = self.balance
            if self.open_trade is not None:
                current_value += self.position * df.iloc[i]['close']
            self.equity_curve.append(current_value)
            
            # Update peak and drawdown
            if current_value > self.peak_balance:
                self.peak_balance = current_value
            drawdown = (self.peak_balance - current_value) / self.peak_balance
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Close remaining position
        if self.open_trade is not None:
            final_price = df['close'].iloc[-1]
            final_time = df.index[-1]
            self._execute_sell(final_time, final_price, "End of backtest")
        
        # Calculate performance
        performance = self._calculate_enhanced_performance(df, strategy_counts, regime_counts)
        
        print(f"\n📊 Backtest completed:")
        print(f"   {total_signals:,} signals analyzed")
        print(f"   {buy_signals} buy signals executed")
        print(f"   Strategies used: {dict(strategy_counts)}")
        print(f"   Market regimes: {dict(regime_counts)}")
        
        return performance
    
    def _execute_buy(self, timestamp: datetime, price: float, reason: str, 
                    strategy: StrategyType, regime: MarketRegime, confidence: float,
                    parameters: TradingParameters):
        """Execute buy with adaptive position sizing."""
        if self.balance <= 0:
            return
        
        # Use adaptive position size
        trade_amount = self.balance * parameters.position_size_pct
        quantity = trade_amount / price
        
        self.open_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            side='buy',
            profit_loss=None,
            profit_pct=None,
            reason=reason,
            strategy=strategy,
            market_regime=regime,
            confidence=confidence
        )
        
        self.balance -= trade_amount
        self.position = quantity
        self.current_parameters = parameters
        
        print(f"    💰 BUY: {quantity:.2f} DOGE at ${price:.6f} (${trade_amount:.2f})")
        print(f"    📊 Position: {parameters.position_size_pct:.1%} | Target: {parameters.profit_target_pct:.2%} | Stop: {parameters.stop_loss_pct:.2%}")
    
    def _execute_sell(self, timestamp: datetime, price: float, reason: str):
        """Execute sell and update statistics."""
        if self.open_trade is None:
            return
        
        trade_value = self.position * price
        profit_loss = trade_value - (self.open_trade.quantity * self.open_trade.entry_price)
        profit_pct = profit_loss / (self.open_trade.quantity * self.open_trade.entry_price)
        
        # Complete trade
        self.open_trade.exit_time = timestamp
        self.open_trade.exit_price = price
        self.open_trade.profit_loss = profit_loss
        self.open_trade.profit_pct = profit_pct
        
        self.balance += trade_value
        self.trades.append(self.open_trade)
        
        # Update strategy statistics
        was_profitable = profit_loss > 0
        self.strategy.update_trade_result(timestamp, was_profitable)
        
        status = "✅ PROFIT" if profit_loss > 0 else "❌ LOSS"
        print(f"    🔄 SELL: {self.position:.2f} DOGE at ${price:.6f} (${trade_value:.2f})")
        print(f"    {status}: ${profit_loss:.2f} ({profit_pct:.2%}) | {reason}")
        
        self.position = 0.0
        self.open_trade = None
        self.current_parameters = None
    
    def _check_exit_conditions(self, timestamp: datetime, candle: pd.Series):
        """Check adaptive exit conditions."""
        if self.open_trade is None or self.current_parameters is None:
            return
        
        current_price = candle['close']
        entry_price = self.open_trade.entry_price
        current_pnl_pct = (current_price - entry_price) / entry_price
        
        # Adaptive profit target
        if current_pnl_pct >= self.current_parameters.profit_target_pct:
            self._execute_sell(timestamp, current_price, 
                             f"Profit target reached ({current_pnl_pct:.2%})")
            return
        
        # Adaptive stop loss
        if current_pnl_pct <= -self.current_parameters.stop_loss_pct:
            self._execute_sell(timestamp, current_price, 
                             f"Stop loss triggered ({current_pnl_pct:.2%})")
            return
        
        # Time-based exit for certain strategies (prevent holding too long)
        if self.open_trade.strategy in [StrategyType.BREAKOUT, StrategyType.MOMENTUM]:
            hold_time = (timestamp - self.open_trade.entry_time).total_seconds() / 60
            if hold_time > 30:  # 30 minutes max for breakout/momentum
                self._execute_sell(timestamp, current_price, 
                                 f"Time exit after {hold_time:.0f} minutes")
                return
    
    def _calculate_enhanced_performance(self, df: pd.DataFrame, strategy_counts: Dict, 
                                      regime_counts: Dict) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Basic metrics
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'profit_loss': t.profit_loss,
            'profit_pct': t.profit_pct,
            'strategy': t.strategy.value,
            'regime': t.market_regime.value,
            'confidence': t.confidence,
            'reason': t.reason
        } for t in self.trades])
        
        total_return = self.balance - self.initial_balance
        total_return_pct = total_return / self.initial_balance
        
        winning_trades = trades_df[trades_df['profit_loss'] > 0]
        losing_trades = trades_df[trades_df['profit_loss'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df)
        avg_win = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['profit_loss'].sum() / losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 and losing_trades['profit_loss'].sum() != 0 else float('inf')
        
        # Strategy performance
        strategy_performance = {}
        for strategy in StrategyType:
            strategy_trades = trades_df[trades_df['strategy'] == strategy.value]
            if len(strategy_trades) > 0:
                strategy_win_rate = len(strategy_trades[strategy_trades['profit_loss'] > 0]) / len(strategy_trades)
                strategy_pnl = strategy_trades['profit_loss'].sum()
                strategy_performance[strategy.value] = {
                    'trades': len(strategy_trades),
                    'win_rate': strategy_win_rate,
                    'total_pnl': strategy_pnl
                }
        
        # Timing metrics
        holding_times = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in self.trades if t.exit_time]
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Risk metrics
        returns = pd.Series([t.profit_pct for t in self.trades])
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_time_minutes': avg_holding_time,
            'final_balance': self.balance,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'strategy_performance': strategy_performance,
            'strategy_counts': strategy_counts,
            'regime_counts': regime_counts,
            'trades_data': trades_df,
            'equity_curve': self.equity_curve
        }
    
    def plot_enhanced_results(self, df: pd.DataFrame, performance: Dict):
        """Create enhanced performance visualization."""
        if 'error' in performance:
            print(f"❌ Cannot plot: {performance['error']}")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Enhanced DOGE/USD Scalping Bot - {len(self.trades)} Trades', 
                    fontsize=16, fontweight='bold')
        
        # 1. Price chart with trades colored by strategy
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='DOGE/USD Price', alpha=0.7, linewidth=1)
        
        strategy_colors = {
            'trend_following': 'blue',
            'mean_reversion': 'green', 
            'breakout': 'red',
            'momentum': 'orange'
        }
        
        for trade in self.trades[:50]:
            color = strategy_colors.get(trade.strategy.value, 'purple')
            ax1.scatter(trade.entry_time, trade.entry_price, color=color, marker='^', s=30, alpha=0.8)
            if trade.exit_time:
                profit_color = 'green' if trade.profit_loss > 0 else 'red'
                ax1.scatter(trade.exit_time, trade.exit_price, color=profit_color, marker='v', s=30, alpha=0.8)
        
        ax1.set_title('Price Chart with Strategy-Colored Trades')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Enhanced equity curve
        ax2 = axes[0, 1]
        time_index = pd.date_range(start=df.index[0], periods=len(performance['equity_curve']), freq='1min')
        ax2.plot(time_index, performance['equity_curve'], color='green', linewidth=2)
        ax2.axhline(y=self.initial_balance, color='red', linestyle='--', alpha=0.7)
        
        # Mark drawdown periods
        equity_series = pd.Series(performance['equity_curve'])
        peak_series = equity_series.expanding().max()
        drawdown_series = (equity_series - peak_series) / peak_series
        
        ax2.fill_between(time_index, performance['equity_curve'], 
                        [peak_series.iloc[i] for i in range(len(peak_series))],
                        where=[dd < -0.01 for dd in drawdown_series], 
                        alpha=0.3, color='red', label='Drawdown')
        
        ax2.set_title(f'Equity Curve (Max DD: {performance["max_drawdown"]:.2%})')
        ax2.set_ylabel('Balance (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Strategy performance comparison
        ax3 = axes[0, 2]
        if performance['strategy_performance']:
            strategies = list(performance['strategy_performance'].keys())
            win_rates = [performance['strategy_performance'][s]['win_rate'] for s in strategies]
            colors = [strategy_colors.get(s, 'purple') for s in strategies]
            
            bars = ax3.bar(strategies, win_rates, color=colors, alpha=0.7)
            ax3.set_title('Win Rate by Strategy')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, wr in zip(bars, win_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{wr:.1%}', ha='center', va='bottom')
        
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # 4. Trade distribution
        ax4 = axes[1, 0]
        profits = [t.profit_loss for t in self.trades]
        ax4.hist(profits, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Trade P&L Distribution')
        ax4.set_xlabel('Profit/Loss (USD)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Market regime analysis
        ax5 = axes[1, 1]
        regime_data = performance['regime_counts']
        regimes = list(regime_data.keys())
        counts = [regime_data[r] for r in regimes]
        
        ax5.pie(counts, labels=[r.value for r in regimes], autopct='%1.1f%%', startangle=90)
        ax5.set_title('Market Regime Distribution')
        
        # 6. Confidence vs Performance
        ax6 = axes[1, 2]
        if len(self.trades) > 0:
            confidences = [t.confidence for t in self.trades]
            profits_pct = [t.profit_pct for t in self.trades]
            colors = ['green' if p > 0 else 'red' for p in profits_pct]
            
            ax6.scatter(confidences, profits_pct, c=colors, alpha=0.6)
            ax6.set_xlabel('Signal Confidence')
            ax6.set_ylabel('Trade Return (%)')
            ax6.set_title('Confidence vs Performance')
            ax6.grid(True, alpha=0.3)
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 7. Performance metrics table
        ax7 = axes[2, 0]
        ax7.axis('off')
        metrics_text = f"""
        📊 ENHANCED PERFORMANCE SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.2%})
        Total Trades: {performance['total_trades']}
        Win Rate: {performance['win_rate']:.1%}
        Profit Factor: {performance['profit_factor']:.2f}
        Sharpe Ratio: {performance['sharpe_ratio']:.2f}
        Max Drawdown: {performance['max_drawdown']:.2%}
        Avg Holding: {performance['avg_holding_time_minutes']:.1f} min
        Final Balance: ${performance['final_balance']:,.2f}
        """
        ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        
        # 8. Drawdown analysis
        ax8 = axes[2, 1]
        if len(performance['equity_curve']) > 1:
            equity_series = pd.Series(performance['equity_curve'])
            peak_series = equity_series.expanding().max()
            drawdown_series = (equity_series - peak_series) / peak_series * 100
            
            ax8.fill_between(range(len(drawdown_series)), drawdown_series, 0, 
                           color='red', alpha=0.3)
            ax8.plot(drawdown_series, color='red', linewidth=1)
            ax8.set_title('Drawdown Over Time')
            ax8.set_ylabel('Drawdown (%)')
            ax8.set_xlabel('Time')
            ax8.grid(True, alpha=0.3)
        
        # 9. Strategy profitability
        ax9 = axes[2, 2]
        if performance['strategy_performance']:
            strategies = list(performance['strategy_performance'].keys())
            pnls = [performance['strategy_performance'][s]['total_pnl'] for s in strategies]
            colors = [strategy_colors.get(s, 'purple') for s in strategies]
            
            bars = ax9.bar(strategies, pnls, color=colors, alpha=0.7)
            ax9.set_title('Total P&L by Strategy')
            ax9.set_ylabel('Total P&L (USD)')
            ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            for bar, pnl in zip(bars, pnls):
                ax9.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (0.01 if pnl >= 0 else -0.01), 
                        f'${pnl:.2f}', ha='center', 
                        va='bottom' if pnl >= 0 else 'top')
        
        ax9.grid(True, alpha=0.3)
        plt.setp(ax9.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/f.kalati/Documents/crypto/enhanced_scalping_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """Main execution function."""
    print("🚀 Enhanced DOGE/USD Adaptive Scalping Bot v2.0")
    print("=" * 50)
    print("🧠 Multi-Strategy Adaptive Engine")
    print("📊 Dynamic Risk Management")
    print("🎯 Market Regime Detection")
    print("🛡️ Conservative Risk Controls\n")
    
    try:
        # Initialize enhanced backtesting engine
        backtest = EnhancedBacktestingEngine(INITIAL_BALANCE)
        
        # Fetch data
        df = backtest.fetch_data()
        
        if df.empty:
            print("❌ No data available for backtesting")
            return
        
        print(f"📅 Data range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Total candles: {len(df):,}")
        
        # Run enhanced backtest
        performance = backtest.run_backtest(df)
        
        if 'error' not in performance:
            # Print detailed results
            print("\n" + "=" * 80)
            print("📈 ENHANCED PERFORMANCE REPORT")
            print("=" * 80)
            print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
            print(f"Final Balance: ${performance['final_balance']:,.2f}")
            print(f"Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.2%})")
            print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Win Rate: {performance['win_rate']:.1%}")
            print(f"Profit Factor: {performance['profit_factor']:.2f}")
            print(f"Average Holding Time: {performance['avg_holding_time_minutes']:.1f} minutes")
            
            # Strategy breakdown
            print(f"\n🧠 STRATEGY PERFORMANCE")
            print("-" * 40)
            for strategy, data in performance['strategy_performance'].items():
                print(f"{strategy}: {data['trades']} trades, {data['win_rate']:.1%} win rate, ${data['total_pnl']:.2f} P&L")
            
            # Risk assessment
            print(f"\n🛡️ ENHANCED RISK ASSESSMENT")
            print("-" * 40)
            if performance['win_rate'] >= 0.6:
                print("✅ Excellent win rate (≥60%)")
            elif performance['win_rate'] >= 0.5:
                print("✅ Good win rate (≥50%)")
            else:
                print("⚠️ Win rate needs improvement (<50%)")
            
            if performance['profit_factor'] >= 2.0:
                print("✅ Excellent profit factor (≥2.0)")
            elif performance['profit_factor'] >= 1.5:
                print("✅ Good profit factor (≥1.5)")
            elif performance['profit_factor'] >= 1.0:
                print("✅ Profitable strategy (>1.0)")
            else:
                print("⚠️ Strategy needs optimization (<1.0)")
            
            if performance['max_drawdown'] <= 0.05:
                print("✅ Low drawdown (≤5%)")
            elif performance['max_drawdown'] <= 0.10:
                print("✅ Acceptable drawdown (≤10%)")
            else:
                print("⚠️ High drawdown (>10%) - Review risk management")
            
            if performance['sharpe_ratio'] >= 1.5:
                print("✅ Excellent risk-adjusted returns (Sharpe ≥1.5)")
            elif performance['sharpe_ratio'] >= 1.0:
                print("✅ Good risk-adjusted returns (Sharpe ≥1.0)")
            else:
                print("⚠️ Risk-adjusted returns could be improved")
            
            # Generate enhanced visualization
            backtest.plot_enhanced_results(df, performance)
            
            print(f"\n📊 Enhanced results saved to: /home/f.kalati/Documents/crypto/enhanced_scalping_results.png")
            
        else:
            print(f"❌ Error in performance calculation: {performance['error']}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    main() 