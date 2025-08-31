#!/usr/bin/env python3
"""
Aggressive DOGE/USD High-Frequency Scalping Bot v3.0
====================================================
High-risk, high-reward scalping bot designed for maximum profits
with aggressive position sizing and frequent trading.

Target: 50%+ monthly returns through high-frequency scalping
Risk: High - suitable for aggressive traders only

WARNING: This bot uses aggressive parameters and high leverage.
Only use with risk capital you can afford to lose completely.

Dependencies:
    pip install ccxt pandas numpy matplotlib seaborn

Usage:
    python aggressive_scalping_bot.py
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
# AGGRESSIVE Configuration
# ---------------------------------------------------------------------------

# Trading Configuration
SYMBOL = "DOGE/USD"
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
TIMEFRAME = "1m"
LOOKBACK_DAYS = 30

# AGGRESSIVE Risk Management (High Risk, High Reward)
INITIAL_BALANCE = 1000.0
AGGRESSIVE_POSITION_SIZE_PCT = 0.25  # 25% per trade (very aggressive)
MAX_POSITION_SIZE_PCT = 0.40         # Up to 40% in volatile markets
MIN_POSITION_SIZE_PCT = 0.15         # Minimum 15% (still aggressive)

# AGGRESSIVE Profit/Loss Targets
AGGRESSIVE_PROFIT_TARGET_PCT = 0.008  # 0.8% profit target (higher)
MAX_PROFIT_TARGET_PCT = 0.015         # 1.5% maximum profit target
MIN_PROFIT_TARGET_PCT = 0.004         # 0.4% minimum profit target

AGGRESSIVE_STOP_LOSS_PCT = 0.012      # 1.2% stop loss (wider)
MAX_STOP_LOSS_PCT = 0.025             # 2.5% maximum stop loss
MIN_STOP_LOSS_PCT = 0.008             # 0.8% minimum stop loss

# Trading Fees (Kraken)
MAKER_FEE = 0.0016  # 0.16% maker fee
TAKER_FEE = 0.0026  # 0.26% taker fee (assume taker for scalping)
TRADING_FEE = TAKER_FEE  # Use taker fee for conservative estimation

# AGGRESSIVE Trading Controls
MAX_TRADES_PER_HOUR = 25              # Much more aggressive
MIN_TIME_BETWEEN_TRADES = 60          # 1 minute only
MAX_CONSECUTIVE_LOSSES = 5            # Allow more losses before stopping

# RELAXED Technical Indicator Parameters
RSI_PERIOD = 10                       # Faster RSI
RSI_OVERSOLD = 45                     # Less extreme
RSI_OVERBOUGHT = 55                   # Less extreme
MACD_FAST = 8                         # Faster MACD
MACD_SLOW = 21
MACD_SIGNAL = 6
BB_PERIOD = 15                        # Shorter BB period
BB_DEVIATION = 1.5                    # Tighter bands
EMA_FAST = 5                          # Much faster EMAs
EMA_SLOW = 13
VOLUME_MA_PERIOD = 10                 # Shorter volume period

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
    AGGRESSIVE_SCALP = "aggressive_scalp"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_EXPANSION = "volatility_expansion"

@dataclass
class AggressiveTradingParameters:
    """Aggressive trading parameters for maximum profits."""
    position_size_pct: float
    profit_target_pct: float
    stop_loss_pct: float
    rsi_oversold: float
    rsi_overbought: float
    min_confidence: float
    strategy_type: StrategyType
    use_leverage: bool = False
    leverage_multiplier: float = 1.0

@dataclass
class Trade:
    """Enhanced trade structure with fees."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str
    gross_profit_loss: Optional[float]
    trading_fees: Optional[float]
    net_profit_loss: Optional[float]
    profit_pct: Optional[float]
    reason: str
    strategy: StrategyType
    market_regime: MarketRegime
    confidence: float

class Signal(NamedTuple):
    """Aggressive trading signal."""
    action: str
    confidence: float
    reason: str
    price: float
    strategy: StrategyType
    parameters: AggressiveTradingParameters

# ---------------------------------------------------------------------------
# Enhanced Technical Indicators
# ---------------------------------------------------------------------------

class AggressiveIndicators:
    """Fast and aggressive technical indicators for high-frequency trading."""
    
    @staticmethod
    def fast_rsi(prices: pd.Series, period: int = 10) -> pd.Series:
        """Fast RSI for aggressive scalping."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def aggressive_signals(prices: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Generate aggressive trading signals."""
        # Price momentum (very short-term)
        momentum_1m = prices.pct_change(1)
        momentum_3m = prices.pct_change(3)
        momentum_5m = prices.pct_change(5)
        
        # Price acceleration
        acceleration = momentum_1m.diff()
        
        # Volume surge detection
        volume_ma = volume.rolling(10).mean()
        volume_surge = volume / volume_ma > 1.3
        
        # Volatility spikes
        volatility = prices.rolling(5).std()
        volatility_spike = volatility > volatility.rolling(20).quantile(0.8)
        
        # Price breakouts (very short-term)
        high_5m = prices.rolling(5).max()
        low_5m = prices.rolling(5).min()
        breakout_up = prices > high_5m.shift(1) * 1.002  # 0.2% breakout
        breakout_down = prices < low_5m.shift(1) * 0.998
        
        return {
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'momentum_5m': momentum_5m,
            'acceleration': acceleration,
            'volume_surge': volume_surge,
            'volatility_spike': volatility_spike,
            'breakout_up': breakout_up,
            'breakout_down': breakout_down,
            'price_above_5ema': prices > prices.ewm(span=5).mean(),
            'volume_above_avg': volume > volume_ma
        }

# ---------------------------------------------------------------------------
# Aggressive Strategy Engine
# ---------------------------------------------------------------------------

class AggressiveStrategy:
    """Ultra-aggressive high-frequency scalping engine."""
    
    def __init__(self):
        self.indicators = AggressiveIndicators()
        self.last_trade_time = None
        self.trades_this_hour = 0
        self.current_hour = None
        self.consecutive_losses = 0
        self.winning_streak = 0
        
    def get_aggressive_parameters(self, confidence: float, volatility_pct: float) -> AggressiveTradingParameters:
        """Get aggressive trading parameters based on market confidence."""
        
        # Base aggressive settings
        base_size = AGGRESSIVE_POSITION_SIZE_PCT
        base_profit = AGGRESSIVE_PROFIT_TARGET_PCT
        base_stop = AGGRESSIVE_STOP_LOSS_PCT
        
        # Increase aggression based on confidence and winning streak
        confidence_multiplier = 1.0 + (confidence - 0.5) * 0.5
        streak_multiplier = 1.0 + min(self.winning_streak * 0.1, 0.3)
        
        # Volatility adjustment
        vol_multiplier = 1.0 + max(volatility_pct - 0.5, 0) * 0.5
        
        total_multiplier = confidence_multiplier * streak_multiplier * vol_multiplier
        
        return AggressiveTradingParameters(
            position_size_pct=min(base_size * total_multiplier, MAX_POSITION_SIZE_PCT),
            profit_target_pct=base_profit * vol_multiplier,
            stop_loss_pct=base_stop * vol_multiplier,
            rsi_oversold=50,  # Very relaxed
            rsi_overbought=50,  # Very relaxed
            min_confidence=0.3,  # Very low threshold
            strategy_type=StrategyType.AGGRESSIVE_SCALP
        )
    
    def analyze_market(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Fast market analysis for aggressive trading."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        analysis = {}
        
        # Fast indicators
        analysis['rsi'] = self.indicators.fast_rsi(close, RSI_PERIOD)
        
        # Fast MACD
        ema_fast = close.ewm(span=MACD_FAST).mean()
        ema_slow = close.ewm(span=MACD_SLOW).mean()
        analysis['macd'] = ema_fast - ema_slow
        analysis['macd_signal'] = analysis['macd'].ewm(span=MACD_SIGNAL).mean()
        analysis['macd_histogram'] = analysis['macd'] - analysis['macd_signal']
        
        # Tight Bollinger Bands
        bb_middle = close.rolling(BB_PERIOD).mean()
        bb_std = close.rolling(BB_PERIOD).std()
        analysis['bb_upper'] = bb_middle + (bb_std * BB_DEVIATION)
        analysis['bb_lower'] = bb_middle - (bb_std * BB_DEVIATION)
        analysis['bb_position'] = (close - analysis['bb_lower']) / (analysis['bb_upper'] - analysis['bb_lower'])
        
        # Fast EMAs
        analysis['ema_fast'] = close.ewm(span=EMA_FAST).mean()
        analysis['ema_slow'] = close.ewm(span=EMA_SLOW).mean()
        analysis['ema_trend'] = analysis['ema_fast'] > analysis['ema_slow']
        
        # Aggressive signals
        aggressive_data = self.indicators.aggressive_signals(close, volume)
        analysis.update(aggressive_data)
        
        # Volume analysis
        analysis['volume_ma'] = volume.rolling(VOLUME_MA_PERIOD).mean()
        analysis['volume_ratio'] = volume / analysis['volume_ma']
        
        # Volatility measure
        analysis['volatility'] = close.rolling(10).std()
        analysis['volatility_pct'] = analysis['volatility'].rolling(50).rank(pct=True)
        
        return analysis
    
    def generate_aggressive_signal(self, df: pd.DataFrame, analysis: Dict[str, pd.Series]) -> Signal:
        """Generate aggressive trading signals with low thresholds."""
        if len(df) < max(RSI_PERIOD, MACD_SLOW, BB_PERIOD, EMA_SLOW):
            return self._hold_signal(df['close'].iloc[-1])
        
        current_price = df['close'].iloc[-1]
        current_time = df.index[-1]
        
        # Relaxed trading restrictions
        if not self._can_trade_aggressive(current_time):
            return self._hold_signal(current_price)
        
        latest_idx = -1
        
        # Get all indicator values
        rsi = analysis['rsi'].iloc[latest_idx] if not pd.isna(analysis['rsi'].iloc[latest_idx]) else 50
        macd_hist = analysis['macd_histogram'].iloc[latest_idx] if not pd.isna(analysis['macd_histogram'].iloc[latest_idx]) else 0
        bb_position = analysis['bb_position'].iloc[latest_idx] if not pd.isna(analysis['bb_position'].iloc[latest_idx]) else 0.5
        volume_ratio = analysis['volume_ratio'].iloc[latest_idx] if not pd.isna(analysis['volume_ratio'].iloc[latest_idx]) else 1.0
        momentum_1m = analysis['momentum_1m'].iloc[latest_idx] if not pd.isna(analysis['momentum_1m'].iloc[latest_idx]) else 0
        momentum_3m = analysis['momentum_3m'].iloc[latest_idx] if not pd.isna(analysis['momentum_3m'].iloc[latest_idx]) else 0
        acceleration = analysis['acceleration'].iloc[latest_idx] if not pd.isna(analysis['acceleration'].iloc[latest_idx]) else 0
        volume_surge = analysis['volume_surge'].iloc[latest_idx] if len(analysis['volume_surge']) > abs(latest_idx) else False
        volatility_spike = analysis['volatility_spike'].iloc[latest_idx] if len(analysis['volatility_spike']) > abs(latest_idx) else False
        ema_trend = analysis['ema_trend'].iloc[latest_idx] if not pd.isna(analysis['ema_trend'].iloc[latest_idx]) else True
        volatility_pct = analysis['volatility_pct'].iloc[latest_idx] if not pd.isna(analysis['volatility_pct'].iloc[latest_idx]) else 0.5
        
        confidence = 0.0
        signals = []
        
        # AGGRESSIVE BUY CONDITIONS (Very Low Threshold)
        
        # Momentum-based entries
        if momentum_1m > 0.001:  # Even tiny positive momentum
            signals.append("Positive 1m momentum")
            confidence += 0.15
        
        if momentum_3m > 0.002:  # Slightly positive 3m momentum
            signals.append("Positive 3m momentum")
            confidence += 0.2
        
        if acceleration > 0:  # Any positive acceleration
            signals.append("Positive acceleration")
            confidence += 0.15
        
        # MACD momentum
        if macd_hist > 0:  # Any positive MACD histogram
            signals.append("MACD momentum building")
            confidence += 0.15
        
        # RSI (very relaxed)
        if 35 <= rsi <= 65:  # Wide RSI range
            signals.append("RSI in tradeable range")
            confidence += 0.1
        
        # Volume confirmation (very low threshold)
        if volume_ratio > 0.8:  # Even below-average volume is OK
            signals.append("Volume present")
            confidence += 0.1
        
        # Bollinger Bands (any position)
        if 0.2 <= bb_position <= 0.8:  # Very wide range
            signals.append("BB position favorable")
            confidence += 0.1
        
        # EMA trend (any trend)
        if ema_trend:
            signals.append("EMA trend positive")
            confidence += 0.1
        
        # Volume surge bonus
        if volume_surge:
            signals.append("Volume surge detected")
            confidence += 0.2
        
        # Volatility spike bonus
        if volatility_spike:
            signals.append("Volatility spike opportunity")
            confidence += 0.2
        
        # Breakout signals
        if len(analysis['breakout_up']) > abs(latest_idx) and analysis['breakout_up'].iloc[latest_idx]:
            signals.append("Price breakout up")
            confidence += 0.25
        
        # Get parameters
        params = self.get_aggressive_parameters(confidence, volatility_pct)
        
        # VERY LOW threshold for entry (aggressive trading)
        if confidence >= params.min_confidence and len(signals) >= 2:
            reason = f"AGGRESSIVE BUY: {', '.join(signals[:4])}"  # Limit reason length
            return Signal('buy', confidence, reason, current_price, params.strategy_type, params)
        
        # Also try some contrarian/mean reversion plays
        contrarian_confidence = 0.0
        contrarian_signals = []
        
        # Contrarian entries on oversold
        if rsi < 40:
            contrarian_signals.append("RSI oversold contrarian")
            contrarian_confidence += 0.2
        
        if bb_position < 0.3:
            contrarian_signals.append("BB lower contrarian")
            contrarian_confidence += 0.2
        
        if momentum_1m < -0.002:  # Negative momentum for reversal
            contrarian_signals.append("Negative momentum reversal")
            contrarian_confidence += 0.15
        
        if volume_ratio > 1.2:  # Volume on the reversal
            contrarian_signals.append("Volume on reversal")
            contrarian_confidence += 0.15
        
        if contrarian_confidence >= 0.4 and len(contrarian_signals) >= 2:
            params.strategy_type = StrategyType.MEAN_REVERSION
            reason = f"CONTRARIAN BUY: {', '.join(contrarian_signals[:3])}"
            return Signal('buy', contrarian_confidence, reason, current_price, params.strategy_type, params)
        
        return self._hold_signal(current_price)
    
    def _hold_signal(self, price: float) -> Signal:
        """Generate hold signal."""
        return Signal('hold', 0.0, 'No clear signal', price, StrategyType.AGGRESSIVE_SCALP, 
                     AggressiveTradingParameters(0.25, 0.008, 0.012, 45, 55, 0.3, StrategyType.AGGRESSIVE_SCALP))
    
    def _can_trade_aggressive(self, current_time: datetime) -> bool:
        """Very relaxed trading restrictions for aggressive trading."""
        current_hour = current_time.hour
        
        # Reset hourly counter
        if self.current_hour != current_hour:
            self.current_hour = current_hour
            self.trades_this_hour = 0
        
        # Much higher hourly limit
        if self.trades_this_hour >= MAX_TRADES_PER_HOUR:
            return False
        
        # Very short minimum time between trades
        if self.last_trade_time:
            time_diff = (current_time - self.last_trade_time).total_seconds()
            if time_diff < MIN_TIME_BETWEEN_TRADES:
                return False
        
        # Allow more consecutive losses
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return False
        
        return True
    
    def update_trade_result(self, trade_time: datetime, was_profitable: bool):
        """Update trading statistics."""
        self.last_trade_time = trade_time
        self.trades_this_hour += 1
        
        if was_profitable:
            self.consecutive_losses = 0
            self.winning_streak += 1
        else:
            self.consecutive_losses += 1
            self.winning_streak = 0

# ---------------------------------------------------------------------------
# Aggressive Backtesting Engine
# ---------------------------------------------------------------------------

class AggressiveBacktestingEngine:
    """High-frequency aggressive backtesting engine."""
    
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0
        self.trades: List[Trade] = []
        self.strategy = AggressiveStrategy()
        self.open_trade: Optional[Trade] = None
        self.current_parameters: Optional[AggressiveTradingParameters] = None
        
        # Performance tracking
        self.total_fees_paid = 0.0
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
    
    def run_aggressive_backtest(self, df: pd.DataFrame) -> Dict:
        """Run aggressive high-frequency backtest."""
        print(f"\n🚀 Starting AGGRESSIVE High-Frequency Backtest")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"⚡ Strategy: Aggressive scalping with {AGGRESSIVE_POSITION_SIZE_PCT:.0%} position size")
        print(f"🎯 Target: {AGGRESSIVE_PROFIT_TARGET_PCT:.1%} profit, {AGGRESSIVE_STOP_LOSS_PCT:.1%} stop")
        print(f"💸 Trading Fee: {TRADING_FEE:.2%} per trade")
        print(f"📈 Max Trades/Hour: {MAX_TRADES_PER_HOUR}")
        print("=" * 70)
        
        # Calculate indicators
        analysis = self.strategy.analyze_market(df)
        
        total_signals = 0
        buy_signals = 0
        strategy_counts = {strategy: 0 for strategy in StrategyType}
        
        # Iterate through data with aggressive scanning
        for i in range(max(RSI_PERIOD, MACD_SLOW, BB_PERIOD, EMA_SLOW), len(df)):
            current_time = df.index[i]
            current_data = df.iloc[:i+1]
            current_analysis = {key: series.iloc[:i+1] for key, series in analysis.items()}
            
            # Generate signal
            signal = self.strategy.generate_aggressive_signal(current_data, current_analysis)
            total_signals += 1
            
            if signal.action != 'hold':
                print(f"🔔 {current_time.strftime('%H:%M')} - {signal.action.upper()}")
                print(f"    Confidence: {signal.confidence:.2f} | {signal.reason[:80]}...")
            
            # Process signal
            if signal.action == 'buy' and self.open_trade is None:
                self._execute_aggressive_buy(current_time, signal.price, signal.reason, 
                                           signal.strategy, signal.confidence, signal.parameters)
                buy_signals += 1
                strategy_counts[signal.strategy] += 1
                
            # Check aggressive exit conditions
            if self.open_trade is not None:
                self._check_aggressive_exit_conditions(current_time, df.iloc[i])
            
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
            self._execute_aggressive_sell(final_time, final_price, "End of backtest")
        
        # Calculate performance
        performance = self._calculate_aggressive_performance(df, strategy_counts)
        
        print(f"\n📊 Aggressive Backtest Completed:")
        print(f"   {total_signals:,} signals analyzed")
        print(f"   {buy_signals} buy signals executed")
        print(f"   Total fees paid: ${self.total_fees_paid:.2f}")
        
        return performance
    
    def _execute_aggressive_buy(self, timestamp: datetime, price: float, reason: str, 
                               strategy: StrategyType, confidence: float,
                               parameters: AggressiveTradingParameters):
        """Execute aggressive buy with fees."""
        if self.balance <= 0:
            return
        
        # Use aggressive position size
        gross_trade_amount = self.balance * parameters.position_size_pct
        entry_fee = gross_trade_amount * TRADING_FEE
        net_trade_amount = gross_trade_amount - entry_fee
        quantity = net_trade_amount / price
        
        self.open_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            side='buy',
            gross_profit_loss=None,
            trading_fees=entry_fee,
            net_profit_loss=None,
            profit_pct=None,
            reason=reason,
            strategy=strategy,
            market_regime=MarketRegime.SIDEWAYS,  # Simplified for now
            confidence=confidence
        )
        
        self.balance -= gross_trade_amount  # Deduct full amount including fees
        self.position = quantity
        self.current_parameters = parameters
        self.total_fees_paid += entry_fee
        
        print(f"    💰 BUY: {quantity:.1f} DOGE at ${price:.6f}")
        print(f"    💸 Amount: ${gross_trade_amount:.2f} (Fee: ${entry_fee:.2f})")
        print(f"    📊 Position: {parameters.position_size_pct:.0%} | Target: {parameters.profit_target_pct:.1%}")
    
    def _execute_aggressive_sell(self, timestamp: datetime, price: float, reason: str):
        """Execute aggressive sell with fees."""
        if self.open_trade is None:
            return
        
        gross_trade_value = self.position * price
        exit_fee = gross_trade_value * TRADING_FEE
        net_trade_value = gross_trade_value - exit_fee
        
        # Calculate P&L
        original_investment = self.open_trade.quantity * self.open_trade.entry_price
        gross_profit_loss = gross_trade_value - original_investment
        total_fees = self.open_trade.trading_fees + exit_fee
        net_profit_loss = gross_profit_loss - exit_fee  # Entry fee already deducted
        profit_pct = net_profit_loss / original_investment
        
        # Complete trade
        self.open_trade.exit_time = timestamp
        self.open_trade.exit_price = price
        self.open_trade.gross_profit_loss = gross_profit_loss
        self.open_trade.trading_fees = total_fees
        self.open_trade.net_profit_loss = net_profit_loss
        self.open_trade.profit_pct = profit_pct
        
        self.balance += net_trade_value
        self.trades.append(self.open_trade)
        self.total_fees_paid += exit_fee
        
        # Update strategy statistics
        was_profitable = net_profit_loss > 0
        self.strategy.update_trade_result(timestamp, was_profitable)
        
        status = "✅ PROFIT" if net_profit_loss > 0 else "❌ LOSS"
        print(f"    🔄 SELL: {self.position:.1f} DOGE at ${price:.6f}")
        print(f"    💰 Net: ${net_trade_value:.2f} (Fee: ${exit_fee:.2f})")
        print(f"    {status}: ${net_profit_loss:.2f} ({profit_pct:.2%}) | {reason[:50]}")
        
        self.position = 0.0
        self.open_trade = None
        self.current_parameters = None
    
    def _check_aggressive_exit_conditions(self, timestamp: datetime, candle: pd.Series):
        """Check aggressive exit conditions with fees consideration."""
        if self.open_trade is None or self.current_parameters is None:
            return
        
        current_price = candle['close']
        entry_price = self.open_trade.entry_price
        
        # Calculate gross P&L percentage
        gross_pnl_pct = (current_price - entry_price) / entry_price
        
        # Account for exit fees in targets
        fee_adjusted_profit_target = self.current_parameters.profit_target_pct + (TRADING_FEE * 2)
        fee_adjusted_stop_loss = -self.current_parameters.stop_loss_pct - (TRADING_FEE * 2)
        
        # Profit target (accounting for fees)
        if gross_pnl_pct >= fee_adjusted_profit_target:
            self._execute_aggressive_sell(timestamp, current_price, 
                                        f"Profit target ({gross_pnl_pct:.2%})")
            return
        
        # Stop loss (accounting for fees)
        if gross_pnl_pct <= fee_adjusted_stop_loss:
            self._execute_aggressive_sell(timestamp, current_price, 
                                        f"Stop loss ({gross_pnl_pct:.2%})")
            return
        
        # Time-based exit for scalping (prevent holding too long)
        hold_time = (timestamp - self.open_trade.entry_time).total_seconds() / 60
        if hold_time > 15:  # Maximum 15 minutes for scalping
            self._execute_aggressive_sell(timestamp, current_price, 
                                        f"Time exit ({hold_time:.0f}min)")
            return
    
    def _calculate_aggressive_performance(self, df: pd.DataFrame, strategy_counts: Dict) -> Dict:
        """Calculate aggressive performance metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Basic metrics
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'gross_profit_loss': t.gross_profit_loss,
            'net_profit_loss': t.net_profit_loss,
            'trading_fees': t.trading_fees,
            'profit_pct': t.profit_pct,
            'strategy': t.strategy.value,
            'confidence': t.confidence,
            'reason': t.reason
        } for t in self.trades])
        
        total_return = self.balance - self.initial_balance
        total_return_pct = total_return / self.initial_balance
        
        # Monthly return projection
        days_in_test = (df.index[-1] - df.index[0]).days
        monthly_return_projection = (total_return_pct * 30 / days_in_test) if days_in_test > 0 else 0
        
        winning_trades = trades_df[trades_df['net_profit_loss'] > 0]
        losing_trades = trades_df[trades_df['net_profit_loss'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df)
        avg_win = winning_trades['net_profit_loss'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_profit_loss'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['net_profit_loss'].sum() / losing_trades['net_profit_loss'].sum()) if len(losing_trades) > 0 and losing_trades['net_profit_loss'].sum() != 0 else float('inf')
        
        # Timing metrics
        holding_times = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in self.trades if t.exit_time]
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Fee analysis
        total_gross_profit = winning_trades['gross_profit_loss'].sum() if len(winning_trades) > 0 else 0
        total_gross_loss = losing_trades['gross_profit_loss'].sum() if len(losing_trades) > 0 else 0
        fee_impact_pct = self.total_fees_paid / self.initial_balance
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'monthly_return_projection': monthly_return_projection,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_time_minutes': avg_holding_time,
            'final_balance': self.balance,
            'max_drawdown': self.max_drawdown,
            'total_fees_paid': self.total_fees_paid,
            'fee_impact_pct': fee_impact_pct,
            'total_gross_profit': total_gross_profit,
            'total_gross_loss': total_gross_loss,
            'strategy_counts': strategy_counts,
            'trades_data': trades_df,
            'equity_curve': self.equity_curve
        }
    
    def plot_aggressive_results(self, df: pd.DataFrame, performance: Dict):
        """Plot aggressive trading results."""
        if 'error' in performance:
            print(f"❌ Cannot plot: {performance['error']}")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Aggressive DOGE/USD Scalping - {len(self.trades)} Trades', 
                    fontsize=16, fontweight='bold')
        
        # 1. Price chart with aggressive trades
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='DOGE/USD Price', alpha=0.7, linewidth=1)
        
        for i, trade in enumerate(self.trades[:100]):  # Show more trades
            color = 'green' if trade.net_profit_loss > 0 else 'red'
            ax1.scatter(trade.entry_time, trade.entry_price, color='blue', marker='^', s=20, alpha=0.7)
            if trade.exit_time:
                ax1.scatter(trade.exit_time, trade.exit_price, color=color, marker='v', s=20, alpha=0.7)
        
        ax1.set_title('Aggressive Scalping Trades')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Aggressive equity curve
        ax2 = axes[0, 1]
        time_index = pd.date_range(start=df.index[0], periods=len(performance['equity_curve']), freq='1min')
        ax2.plot(time_index, performance['equity_curve'], color='green', linewidth=2)
        ax2.axhline(y=self.initial_balance, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_title(f'Equity Curve (Target: 50%+ monthly)')
        ax2.set_ylabel('Balance (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade frequency over time
        ax3 = axes[0, 2]
        trade_times = [t.entry_time for t in self.trades]
        trade_hours = [t.hour for t in trade_times]
        ax3.hist(trade_hours, bins=24, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Trade Frequency by Hour')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Number of Trades')
        ax3.grid(True, alpha=0.3)
        
        # 4. P&L distribution with fees
        ax4 = axes[1, 0]
        net_profits = [t.net_profit_loss for t in self.trades]
        gross_profits = [t.gross_profit_loss for t in self.trades]
        
        ax4.hist(gross_profits, bins=15, alpha=0.5, label='Gross P&L', color='blue')
        ax4.hist(net_profits, bins=15, alpha=0.7, label='Net P&L (after fees)', color='green')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('P&L Distribution (Gross vs Net)')
        ax4.set_xlabel('Profit/Loss (USD)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics
        ax5 = axes[1, 1]
        ax5.axis('off')
        metrics_text = f"""
        🚀 AGGRESSIVE PERFORMANCE SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.1%})
        Monthly Projection: {performance['monthly_return_projection']:.1%}
        Total Trades: {performance['total_trades']}
        Win Rate: {performance['win_rate']:.1%}
        Profit Factor: {performance['profit_factor']:.2f}
        Max Drawdown: {performance['max_drawdown']:.1%}
        
        💸 FEE ANALYSIS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Fees: ${performance['total_fees_paid']:.2f}
        Fee Impact: {performance['fee_impact_pct']:.2%} of capital
        Avg Hold Time: {performance['avg_holding_time_minutes']:.1f} min
        
        🎯 TARGET ANALYSIS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Target: 50%+ monthly return
        Achieved: {performance['monthly_return_projection']:.1%} monthly
        Status: {'✅ TARGET MET' if performance['monthly_return_projection'] >= 0.5 else '❌ BELOW TARGET'}
        """
        ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace')
        
        # 6. Fee impact analysis
        ax6 = axes[1, 2]
        if len(self.trades) > 0:
            trade_numbers = list(range(1, len(self.trades) + 1))
            cumulative_fees = np.cumsum([t.trading_fees for t in self.trades])
            cumulative_gross = np.cumsum([t.gross_profit_loss for t in self.trades])
            cumulative_net = np.cumsum([t.net_profit_loss for t in self.trades])
            
            ax6.plot(trade_numbers, cumulative_gross, label='Gross P&L', color='blue', alpha=0.7)
            ax6.plot(trade_numbers, cumulative_net, label='Net P&L', color='green', linewidth=2)
            ax6.plot(trade_numbers, -cumulative_fees, label='Cumulative Fees', color='red', linestyle='--')
            
            ax6.set_title('Cumulative P&L vs Fees')
            ax6.set_xlabel('Trade Number')
            ax6.set_ylabel('Cumulative Amount (USD)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/f.kalati/Documents/crypto/aggressive_scalping_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """Main execution function for aggressive scalping."""
    print("🚀 AGGRESSIVE DOGE/USD High-Frequency Scalping Bot v3.0")
    print("=" * 60)
    print("⚡ HIGH RISK - HIGH REWARD")
    print("🎯 Target: 50%+ Monthly Returns")
    print("💸 Includes Trading Fees")
    print("🔥 Maximum Aggression Mode\n")
    
    print("⚠️  WARNING: This bot uses aggressive parameters!")
    print("   Only use with risk capital you can afford to lose completely.\n")
    
    try:
        # Initialize aggressive backtesting engine
        backtest = AggressiveBacktestingEngine(INITIAL_BALANCE)
        
        # Fetch data
        df = backtest.fetch_data()
        
        if df.empty:
            print("❌ No data available for backtesting")
            return
        
        print(f"📅 Data range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Total candles: {len(df):,}")
        
        # Run aggressive backtest
        performance = backtest.run_aggressive_backtest(df)
        
        if 'error' not in performance:
            # Print detailed results
            print("\n" + "=" * 80)
            print("🚀 AGGRESSIVE PERFORMANCE REPORT")
            print("=" * 80)
            print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
            print(f"Final Balance: ${performance['final_balance']:,.2f}")
            print(f"Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.2%})")
            print(f"📈 MONTHLY PROJECTION: {performance['monthly_return_projection']:.1%}")
            print(f"🎯 Target (50% monthly): {'✅ ACHIEVED' if performance['monthly_return_projection'] >= 50 else '❌ NOT ACHIEVED'}")
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Win Rate: {performance['win_rate']:.1%}")
            print(f"Profit Factor: {performance['profit_factor']:.2f}")
            print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
            
            # Fee analysis
            print(f"\n💸 FEE IMPACT ANALYSIS")
            print("-" * 40)
            print(f"Total Fees Paid: ${performance['total_fees_paid']:.2f}")
            print(f"Fee Impact: {performance['fee_impact_pct']:.2%} of capital")
            print(f"Average Holding Time: {performance['avg_holding_time_minutes']:.1f} minutes")
            
            # Risk assessment
            print(f"\n🛡️ RISK ASSESSMENT")
            print("-" * 40)
            if performance['monthly_return_projection'] >= 50:
                print("🎯 TARGET ACHIEVED: 50%+ monthly return projection")
            else:
                print(f"⚠️ BELOW TARGET: {performance['monthly_return_projection']:.1%} monthly (target: 50%+)")
            
            if performance['total_trades'] >= 50:
                print("✅ High trading frequency achieved")
            elif performance['total_trades'] >= 20:
                print("✅ Good trading frequency")
            else:
                print("⚠️ Low trading frequency - consider more aggressive parameters")
            
            if performance['win_rate'] >= 0.6:
                print("✅ Excellent win rate")
            elif performance['win_rate'] >= 0.5:
                print("✅ Good win rate")
            else:
                print("⚠️ Win rate could be improved")
            
            # Generate visualization
            backtest.plot_aggressive_results(df, performance)
            
            print(f"\n📊 Aggressive results saved to: /home/f.kalati/Documents/crypto/aggressive_scalping_results.png")
            
            # Final recommendation
            print(f"\n💡 FINAL ASSESSMENT")
            print("-" * 40)
            if performance['monthly_return_projection'] >= 50 and performance['total_trades'] >= 20:
                print("🎉 SUCCESS: Aggressive strategy achieved target!")
                print("   Consider implementing with proper risk management.")
            elif performance['monthly_return_projection'] >= 25:
                print("📈 PARTIAL SUCCESS: Good returns but below 50% target.")
                print("   Consider increasing aggression or different timeframes.")
            else:
                print("🔄 NEEDS OPTIMIZATION: Returns below expectations.")
                print("   Strategy needs further refinement for target achievement.")
            
        else:
            print(f"❌ Error in performance calculation: {performance['error']}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    main() 