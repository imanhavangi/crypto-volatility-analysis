#!/usr/bin/env python3
"""
ULTRA AGGRESSIVE DOGE/USD Scalping Bot v4.0 - EXTREME RISK
==========================================================
MAXIMUM RISK - MAXIMUM REWARD
Target: 50%+ monthly returns through ultra-high frequency trading
with massive position sizes and simulated leverage.

⚠️ EXTREME WARNING: This bot uses DANGEROUS parameters!
- Uses up to 50% position sizes per trade
- Simulates 2x leverage for maximum aggression
- Extremely relaxed entry conditions
- Target: 100+ trades per month

ONLY use with money you can afford to lose COMPLETELY!

Dependencies:
    pip install ccxt pandas numpy matplotlib seaborn

Usage:
    python ultra_aggressive_scalping_bot.py
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
# ULTRA AGGRESSIVE Configuration - EXTREME RISK
# ---------------------------------------------------------------------------

# Trading Configuration
SYMBOL = "DOGE/USD"
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
TIMEFRAME = "1m"
LOOKBACK_DAYS = 30

# ULTRA AGGRESSIVE Risk Management (EXTREME RISK)
INITIAL_BALANCE = 1000.0
ULTRA_AGGRESSIVE_POSITION_SIZE_PCT = 0.40  # 40% per trade (EXTREME)
MAX_POSITION_SIZE_PCT = 0.60               # Up to 60% in high confidence
MIN_POSITION_SIZE_PCT = 0.25               # Minimum 25% (still extreme)

# Leverage Simulation (2x for maximum aggression)
USE_SIMULATED_LEVERAGE = True
LEVERAGE_MULTIPLIER = 2.0

# ULTRA AGGRESSIVE Profit/Loss Targets
ULTRA_PROFIT_TARGET_PCT = 0.005   # 0.5% profit target (smaller for more trades)
MAX_PROFIT_TARGET_PCT = 0.012     # 1.2% maximum profit target
MIN_PROFIT_TARGET_PCT = 0.003     # 0.3% minimum profit target

ULTRA_STOP_LOSS_PCT = 0.008       # 0.8% stop loss (smaller for more trades)
MAX_STOP_LOSS_PCT = 0.020         # 2.0% maximum stop loss
MIN_STOP_LOSS_PCT = 0.005         # 0.5% minimum stop loss

# Trading Fees (reduced for calculation)
MAKER_FEE = 0.0010  # Assume maker fee for aggressive scalping
TAKER_FEE = 0.0015  # Reduced taker fee assumption
TRADING_FEE = MAKER_FEE  # Use maker fee for optimistic calculation

# ULTRA AGGRESSIVE Trading Controls
MAX_TRADES_PER_HOUR = 50              # EXTREME frequency
MIN_TIME_BETWEEN_TRADES = 30          # 30 seconds only
MAX_CONSECUTIVE_LOSSES = 8            # Allow more losses
PROFIT_REINVESTMENT = True            # Compound profits

# EXTREMELY RELAXED Technical Indicator Parameters
RSI_PERIOD = 7                        # Ultra-fast RSI
RSI_OVERSOLD = 50                     # No real bounds
RSI_OVERBOUGHT = 50                   # No real bounds
MACD_FAST = 5                         # Ultra-fast MACD
MACD_SLOW = 13
MACD_SIGNAL = 3
BB_PERIOD = 10                        # Very short BB period
BB_DEVIATION = 1.0                    # Very tight bands
EMA_FAST = 3                          # Ultra-fast EMAs
EMA_SLOW = 8
VOLUME_MA_PERIOD = 5                  # Very short volume period

# Ultra-aggressive confidence threshold
MIN_CONFIDENCE = 0.1                  # Almost no threshold

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class UltraAggressiveTradingParameters:
    """Ultra-aggressive trading parameters for maximum profits."""
    position_size_pct: float
    profit_target_pct: float
    stop_loss_pct: float
    leverage_multiplier: float
    min_confidence: float

@dataclass
class Trade:
    """Trade structure with leverage simulation."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    effective_quantity: float  # With leverage
    side: str
    gross_profit_loss: Optional[float]
    trading_fees: Optional[float]
    net_profit_loss: Optional[float]
    profit_pct: Optional[float]
    reason: str
    confidence: float
    leverage_used: float

class Signal(NamedTuple):
    """Ultra-aggressive trading signal."""
    action: str
    confidence: float
    reason: str
    price: float
    parameters: UltraAggressiveTradingParameters

# ---------------------------------------------------------------------------
# Ultra-Fast Indicators
# ---------------------------------------------------------------------------

class UltraFastIndicators:
    """Ultra-fast indicators for extreme high-frequency trading."""
    
    @staticmethod
    def ultra_fast_signals(prices: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Generate ultra-fast trading signals with minimal calculation."""
        
        # Ultra-short momentum
        momentum_30s = prices.pct_change(1)
        momentum_60s = prices.pct_change(2)
        
        # Price direction (any movement)
        price_up = momentum_30s > 0
        price_down = momentum_30s < 0
        
        # Volume activity (any volume)
        volume_active = volume > 0
        
        # Price oscillations (for quick scalps)
        price_change_abs = abs(momentum_30s)
        micro_volatility = price_change_abs > 0.0005  # 0.05% moves
        
        # Ultra-simple RSI (last 5 periods)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=5, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=5, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        ultra_rsi = 100 - (100 / (1 + rs))
        
        # ANY signal is a trade signal in ultra-aggressive mode
        any_momentum = abs(momentum_30s) > 0.0001  # Any tiny movement
        any_volume = volume > volume.rolling(3).mean() * 0.5  # Half average volume
        
        return {
            'momentum_30s': momentum_30s,
            'momentum_60s': momentum_60s,
            'price_up': price_up,
            'price_down': price_down,
            'volume_active': volume_active,
            'micro_volatility': micro_volatility,
            'ultra_rsi': ultra_rsi,
            'any_momentum': any_momentum,
            'any_volume': any_volume,
            'always_true': pd.Series([True] * len(prices), index=prices.index)  # Always trade!
        }

# ---------------------------------------------------------------------------
# Ultra-Aggressive Strategy Engine
# ---------------------------------------------------------------------------

class UltraAggressiveStrategy:
    """Ultra-aggressive strategy with minimal restrictions."""
    
    def __init__(self):
        self.indicators = UltraFastIndicators()
        self.last_trade_time = None
        self.trades_this_hour = 0
        self.current_hour = None
        self.consecutive_losses = 0
        self.winning_streak = 0
        self.total_trades = 0
        
    def get_ultra_aggressive_parameters(self, confidence: float) -> UltraAggressiveTradingParameters:
        """Get ultra-aggressive parameters - always maximum aggression."""
        
        # Always use maximum aggression
        base_size = ULTRA_AGGRESSIVE_POSITION_SIZE_PCT
        
        # Increase based on confidence and winning streak
        confidence_boost = confidence * 0.2
        streak_boost = min(self.winning_streak * 0.05, 0.15)
        
        position_size = min(base_size + confidence_boost + streak_boost, MAX_POSITION_SIZE_PCT)
        
        return UltraAggressiveTradingParameters(
            position_size_pct=position_size,
            profit_target_pct=ULTRA_PROFIT_TARGET_PCT,
            stop_loss_pct=ULTRA_STOP_LOSS_PCT,
            leverage_multiplier=LEVERAGE_MULTIPLIER if USE_SIMULATED_LEVERAGE else 1.0,
            min_confidence=MIN_CONFIDENCE
        )
    
    def analyze_market_ultra_fast(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Ultra-fast market analysis with minimal indicators."""
        close = df['close']
        volume = df['volume']
        
        # Only essential ultra-fast signals
        analysis = self.indicators.ultra_fast_signals(close, volume)
        
        # Add minimal additional data
        analysis['volume_ma'] = volume.rolling(VOLUME_MA_PERIOD).mean()
        analysis['price_sma'] = close.rolling(EMA_FAST).mean()
        
        return analysis
    
    def generate_ultra_aggressive_signal(self, df: pd.DataFrame, analysis: Dict[str, pd.Series]) -> Signal:
        """Generate ultra-aggressive signals - trade on ANYTHING."""
        if len(df) < 10:  # Minimal data requirement
            return self._hold_signal(df['close'].iloc[-1])
        
        current_price = df['close'].iloc[-1]
        current_time = df.index[-1]
        
        # Ultra-relaxed trading restrictions
        if not self._can_trade_ultra_aggressive(current_time):
            return self._hold_signal(current_price)
        
        latest_idx = -1
        
        # Get ultra-fast signals
        momentum_30s = analysis['momentum_30s'].iloc[latest_idx] if len(analysis['momentum_30s']) > abs(latest_idx) else 0
        price_up = analysis['price_up'].iloc[latest_idx] if len(analysis['price_up']) > abs(latest_idx) else False
        volume_active = analysis['volume_active'].iloc[latest_idx] if len(analysis['volume_active']) > abs(latest_idx) else False
        micro_volatility = analysis['micro_volatility'].iloc[latest_idx] if len(analysis['micro_volatility']) > abs(latest_idx) else False
        ultra_rsi = analysis['ultra_rsi'].iloc[latest_idx] if not pd.isna(analysis['ultra_rsi'].iloc[latest_idx]) else 50
        any_momentum = analysis['any_momentum'].iloc[latest_idx] if len(analysis['any_momentum']) > abs(latest_idx) else False
        any_volume = analysis['any_volume'].iloc[latest_idx] if len(analysis['any_volume']) > abs(latest_idx) else False
        
        confidence = 0.0
        signals = []
        
        # ULTRA-AGGRESSIVE BUY CONDITIONS - Trade on ANYTHING!
        
        # ANY price movement
        if any_momentum:
            signals.append("Price movement detected")
            confidence += 0.3
        
        # ANY upward movement
        if price_up:
            signals.append("Price moving up")
            confidence += 0.4
        
        # ANY volume activity
        if any_volume:
            signals.append("Volume activity")
            confidence += 0.2
        
        # ANY micro volatility
        if micro_volatility:
            signals.append("Micro volatility")
            confidence += 0.3
        
        # RSI in any range (always trade)
        if 20 <= ultra_rsi <= 80:  # Almost always true
            signals.append("RSI in range")
            confidence += 0.2
        
        # Positive momentum bonus
        if momentum_30s > 0:
            signals.append("Positive momentum")
            confidence += 0.5
        
        # Volume active bonus
        if volume_active:
            signals.append("Volume active")
            confidence += 0.2
        
        # ALWAYS TRADE MODE - if no signals, create one
        if len(signals) == 0:
            signals.append("Always trade mode")
            confidence = 0.5
        
        # Get ultra-aggressive parameters
        params = self.get_ultra_aggressive_parameters(confidence)
        
        # TRADE ON ALMOST ANYTHING
        if confidence >= params.min_confidence or len(signals) >= 1:
            reason = f"ULTRA BUY: {', '.join(signals[:3])}"
            return Signal('buy', confidence, reason, current_price, params)
        
        return self._hold_signal(current_price)
    
    def _hold_signal(self, price: float) -> Signal:
        """Generate hold signal."""
        params = UltraAggressiveTradingParameters(0.4, 0.005, 0.008, 2.0, 0.1)
        return Signal('hold', 0.0, 'No signal', price, params)
    
    def _can_trade_ultra_aggressive(self, current_time: datetime) -> bool:
        """Ultra-relaxed trading restrictions."""
        current_hour = current_time.hour
        
        # Reset hourly counter
        if self.current_hour != current_hour:
            self.current_hour = current_hour
            self.trades_this_hour = 0
        
        # Ultra-high hourly limit
        if self.trades_this_hour >= MAX_TRADES_PER_HOUR:
            return False
        
        # Ultra-short minimum time between trades
        if self.last_trade_time:
            time_diff = (current_time - self.last_trade_time).total_seconds()
            if time_diff < MIN_TIME_BETWEEN_TRADES:
                return False
        
        # Allow many consecutive losses
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return False
        
        return True
    
    def update_trade_result(self, trade_time: datetime, was_profitable: bool):
        """Update trading statistics."""
        self.last_trade_time = trade_time
        self.trades_this_hour += 1
        self.total_trades += 1
        
        if was_profitable:
            self.consecutive_losses = 0
            self.winning_streak += 1
        else:
            self.consecutive_losses += 1
            self.winning_streak = 0

# ---------------------------------------------------------------------------
# Ultra-Aggressive Backtesting Engine
# ---------------------------------------------------------------------------

class UltraAggressiveBacktestingEngine:
    """Ultra-aggressive backtesting with leverage simulation."""
    
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0
        self.effective_position = 0.0  # With leverage
        self.trades: List[Trade] = []
        self.strategy = UltraAggressiveStrategy()
        self.open_trade: Optional[Trade] = None
        self.current_parameters: Optional[UltraAggressiveTradingParameters] = None
        
        # Performance tracking
        self.total_fees_paid = 0.0
        self.equity_curve = [initial_balance]
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.total_volume_traded = 0.0
        
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
    
    def run_ultra_aggressive_backtest(self, df: pd.DataFrame) -> Dict:
        """Run ultra-aggressive high-frequency backtest with leverage."""
        print(f"\n🚀 Starting ULTRA AGGRESSIVE Backtest - EXTREME RISK")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"🔥 Strategy: ULTRA aggressive with {ULTRA_AGGRESSIVE_POSITION_SIZE_PCT:.0%} position size")
        print(f"⚡ Leverage: {LEVERAGE_MULTIPLIER}x simulated")
        print(f"🎯 Target: {ULTRA_PROFIT_TARGET_PCT:.1%} profit, {ULTRA_STOP_LOSS_PCT:.1%} stop")
        print(f"💸 Trading Fee: {TRADING_FEE:.2%} per trade")
        print(f"📈 Max Trades/Hour: {MAX_TRADES_PER_HOUR}")
        print(f"🔄 Min Trade Interval: {MIN_TIME_BETWEEN_TRADES}s")
        print("=" * 70)
        
        # Calculate ultra-fast indicators
        analysis = self.strategy.analyze_market_ultra_fast(df)
        
        total_signals = 0
        buy_signals = 0
        
        # Iterate through data with ultra-aggressive scanning
        for i in range(10, len(df)):  # Start after minimal indicators
            current_time = df.index[i]
            current_data = df.iloc[:i+1]
            current_analysis = {key: series.iloc[:i+1] for key, series in analysis.items()}
            
            # Generate signal
            signal = self.strategy.generate_ultra_aggressive_signal(current_data, current_analysis)
            total_signals += 1
            
            if signal.action != 'hold':
                print(f"🔔 {current_time.strftime('%H:%M')} - ULTRA BUY")
                print(f"    C: {signal.confidence:.1f} | {signal.reason[:60]}...")
            
            # Process signal
            if signal.action == 'buy' and self.open_trade is None:
                self._execute_ultra_aggressive_buy(current_time, signal.price, signal.reason, 
                                                 signal.confidence, signal.parameters)
                buy_signals += 1
                
            # Check ultra-aggressive exit conditions
            if self.open_trade is not None:
                self._check_ultra_aggressive_exit_conditions(current_time, df.iloc[i])
            
            # Update equity curve (including leverage effects)
            current_value = self.balance
            if self.open_trade is not None:
                # Calculate unrealized P&L with leverage
                current_price = df.iloc[i]['close']
                unrealized_pnl = (current_price - self.open_trade.entry_price) * self.open_trade.effective_quantity
                current_value += unrealized_pnl
            
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
            self._execute_ultra_aggressive_sell(final_time, final_price, "End of backtest")
        
        # Calculate performance
        performance = self._calculate_ultra_aggressive_performance(df)
        
        print(f"\n📊 ULTRA Aggressive Backtest Completed:")
        print(f"   {total_signals:,} signals analyzed")
        print(f"   {buy_signals} ULTRA buy signals executed")
        print(f"   Total fees paid: ${self.total_fees_paid:.2f}")
        print(f"   Total volume traded: ${self.total_volume_traded:,.0f}")
        
        return performance
    
    def _execute_ultra_aggressive_buy(self, timestamp: datetime, price: float, reason: str, 
                                    confidence: float, parameters: UltraAggressiveTradingParameters):
        """Execute ultra-aggressive buy with leverage simulation."""
        if self.balance <= 0:
            return
        
        # Use ultra-aggressive position size
        gross_trade_amount = self.balance * parameters.position_size_pct
        entry_fee = gross_trade_amount * TRADING_FEE
        net_trade_amount = gross_trade_amount - entry_fee
        quantity = net_trade_amount / price
        
        # Apply leverage simulation
        effective_quantity = quantity * parameters.leverage_multiplier
        
        self.open_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            effective_quantity=effective_quantity,
            side='buy',
            gross_profit_loss=None,
            trading_fees=entry_fee,
            net_profit_loss=None,
            profit_pct=None,
            reason=reason,
            confidence=confidence,
            leverage_used=parameters.leverage_multiplier
        )
        
        self.balance -= gross_trade_amount
        self.position = quantity
        self.effective_position = effective_quantity
        self.current_parameters = parameters
        self.total_fees_paid += entry_fee
        self.total_volume_traded += gross_trade_amount
        
        print(f"    💰 BUY: {quantity:.0f} DOGE (Effective: {effective_quantity:.0f}) at ${price:.6f}")
        print(f"    💸 Amount: ${gross_trade_amount:.2f} (Fee: ${entry_fee:.2f})")
        print(f"    ⚡ Leverage: {parameters.leverage_multiplier}x | Size: {parameters.position_size_pct:.0%}")
    
    def _execute_ultra_aggressive_sell(self, timestamp: datetime, price: float, reason: str):
        """Execute ultra-aggressive sell with leverage effects."""
        if self.open_trade is None:
            return
        
        # Calculate P&L with leverage effects
        price_change = price - self.open_trade.entry_price
        gross_profit_loss = price_change * self.open_trade.effective_quantity  # Leverage effect
        
        # Calculate trade value and fees
        gross_trade_value = self.open_trade.quantity * price  # Base quantity for fees
        exit_fee = gross_trade_value * TRADING_FEE
        net_trade_value = gross_trade_value - exit_fee
        
        # Net P&L after fees
        total_fees = self.open_trade.trading_fees + exit_fee
        net_profit_loss = gross_profit_loss - exit_fee
        
        # Profit percentage based on effective investment
        original_investment = self.open_trade.quantity * self.open_trade.entry_price
        profit_pct = net_profit_loss / original_investment
        
        # Complete trade
        self.open_trade.exit_time = timestamp
        self.open_trade.exit_price = price
        self.open_trade.gross_profit_loss = gross_profit_loss
        self.open_trade.trading_fees = total_fees
        self.open_trade.net_profit_loss = net_profit_loss
        self.open_trade.profit_pct = profit_pct
        
        # Update balance with leverage effects
        self.balance += net_trade_value + gross_profit_loss - exit_fee
        
        # Compound profits if enabled
        if PROFIT_REINVESTMENT and net_profit_loss > 0:
            # Profits are automatically reinvested by adding to balance
            pass
        
        self.trades.append(self.open_trade)
        self.total_fees_paid += exit_fee
        self.total_volume_traded += gross_trade_value
        
        # Update strategy statistics
        was_profitable = net_profit_loss > 0
        self.strategy.update_trade_result(timestamp, was_profitable)
        
        status = "✅ PROFIT" if net_profit_loss > 0 else "❌ LOSS"
        print(f"    🔄 SELL: {self.position:.0f} DOGE at ${price:.6f}")
        print(f"    💰 Net: ${net_trade_value:.2f} (Fee: ${exit_fee:.2f})")
        print(f"    {status}: ${net_profit_loss:.2f} ({profit_pct:.2%}) | {reason[:30]}")
        
        self.position = 0.0
        self.effective_position = 0.0
        self.open_trade = None
        self.current_parameters = None
    
    def _check_ultra_aggressive_exit_conditions(self, timestamp: datetime, candle: pd.Series):
        """Check ultra-aggressive exit conditions."""
        if self.open_trade is None or self.current_parameters is None:
            return
        
        current_price = candle['close']
        entry_price = self.open_trade.entry_price
        
        # Calculate P&L percentage with leverage effects
        price_change_pct = (current_price - entry_price) / entry_price
        leveraged_pnl_pct = price_change_pct * self.open_trade.leverage_used
        
        # Account for fees in targets
        fee_adjusted_profit_target = self.current_parameters.profit_target_pct + (TRADING_FEE * 2)
        fee_adjusted_stop_loss = -self.current_parameters.stop_loss_pct - (TRADING_FEE * 2)
        
        # Profit target (with leverage)
        if leveraged_pnl_pct >= fee_adjusted_profit_target:
            self._execute_ultra_aggressive_sell(timestamp, current_price, 
                                              f"Profit target ({leveraged_pnl_pct:.2%})")
            return
        
        # Stop loss (with leverage)
        if leveraged_pnl_pct <= fee_adjusted_stop_loss:
            self._execute_ultra_aggressive_sell(timestamp, current_price, 
                                              f"Stop loss ({leveraged_pnl_pct:.2%})")
            return
        
        # Ultra-fast time exit (scalping)
        hold_time = (timestamp - self.open_trade.entry_time).total_seconds() / 60
        if hold_time > 10:  # Maximum 10 minutes for ultra-scalping
            self._execute_ultra_aggressive_sell(timestamp, current_price, 
                                              f"Time exit ({hold_time:.0f}min)")
            return
    
    def _calculate_ultra_aggressive_performance(self, df: pd.DataFrame) -> Dict:
        """Calculate ultra-aggressive performance metrics."""
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
            'leverage_used': t.leverage_used,
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
        
        # Ultra-aggressive specific metrics
        total_leverage_exposure = sum(t.effective_quantity * t.entry_price for t in self.trades)
        avg_leverage = np.mean([t.leverage_used for t in self.trades])
        
        # Fee analysis
        fee_impact_pct = self.total_fees_paid / self.initial_balance
        volume_turnover = self.total_volume_traded / self.initial_balance
        
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
            'total_volume_traded': self.total_volume_traded,
            'volume_turnover': volume_turnover,
            'total_leverage_exposure': total_leverage_exposure,
            'avg_leverage': avg_leverage,
            'trades_data': trades_df,
            'equity_curve': self.equity_curve
        }
    
    def plot_ultra_aggressive_results(self, df: pd.DataFrame, performance: Dict):
        """Plot ultra-aggressive results with leverage visualization."""
        if 'error' in performance:
            print(f"❌ Cannot plot: {performance['error']}")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'ULTRA AGGRESSIVE DOGE/USD - {len(self.trades)} Trades with {performance["avg_leverage"]:.1f}x Avg Leverage', 
                    fontsize=16, fontweight='bold')
        
        # 1. Price chart with ultra-aggressive trades
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='DOGE/USD Price', alpha=0.7, linewidth=1)
        
        for i, trade in enumerate(self.trades[:200]):  # Show many trades
            color = 'green' if trade.net_profit_loss > 0 else 'red'
            alpha = min(0.8, trade.leverage_used / 3.0)  # Alpha based on leverage
            ax1.scatter(trade.entry_time, trade.entry_price, color='blue', marker='^', s=15, alpha=alpha)
            if trade.exit_time:
                ax1.scatter(trade.exit_time, trade.exit_price, color=color, marker='v', s=15, alpha=alpha)
        
        ax1.set_title('Ultra-Aggressive Trades (Size by Leverage)')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Equity curve with target line
        ax2 = axes[0, 1]
        time_index = pd.date_range(start=df.index[0], periods=len(performance['equity_curve']), freq='1min')
        ax2.plot(time_index, performance['equity_curve'], color='green', linewidth=2, label='Actual')
        
        # Target 50% monthly line
        target_50_pct = self.initial_balance * 1.5
        ax2.axhline(y=target_50_pct, color='red', linestyle='--', alpha=0.7, label='50% Target')
        ax2.axhline(y=self.initial_balance, color='blue', linestyle='--', alpha=0.7, label='Break-even')
        
        ax2.set_title(f'Equity Curve - Target: 50% Monthly')
        ax2.set_ylabel('Balance (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade frequency and volume
        ax3 = axes[0, 2]
        if len(self.trades) > 0:
            trade_times = [t.entry_time for t in self.trades]
            trade_hours = [t.hour for t in trade_times]
            ax3.hist(trade_hours, bins=24, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title(f'Trade Frequency by Hour ({len(self.trades)} total)')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Number of Trades')
            ax3.grid(True, alpha=0.3)
        
        # 4. Leverage impact analysis
        ax4 = axes[1, 0]
        if len(self.trades) > 0:
            leverages = [t.leverage_used for t in self.trades]
            profits = [t.net_profit_loss for t in self.trades]
            colors = ['green' if p > 0 else 'red' for p in profits]
            
            ax4.scatter(leverages, profits, c=colors, alpha=0.6)
            ax4.set_xlabel('Leverage Used')
            ax4.set_ylabel('Trade P&L (USD)')
            ax4.set_title('Leverage vs P&L')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 5. Performance metrics
        ax5 = axes[1, 1]
        ax5.axis('off')
        
        target_met = "✅ TARGET MET" if performance['monthly_return_projection'] >= 0.5 else "❌ TARGET MISSED"
        target_color = "green" if performance['monthly_return_projection'] >= 0.5 else "red"
        
        metrics_text = f"""
        🚀 ULTRA AGGRESSIVE PERFORMANCE
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.1%})
        Monthly Projection: {performance['monthly_return_projection']:.1%}
        {target_met}
        
        📊 TRADING STATS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Trades: {performance['total_trades']}
        Win Rate: {performance['win_rate']:.1%}
        Profit Factor: {performance['profit_factor']:.2f}
        Avg Leverage: {performance['avg_leverage']:.1f}x
        
        💸 RISK METRICS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Max Drawdown: {performance['max_drawdown']:.1%}
        Total Fees: ${performance['total_fees_paid']:.2f}
        Volume Turnover: {performance['volume_turnover']:.1f}x
        Avg Hold Time: {performance['avg_holding_time_minutes']:.1f} min
        """
        ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                color=target_color if 'TARGET' in metrics_text else 'black')
        
        # 6. Cumulative volume and returns
        ax6 = axes[1, 2]
        if len(self.trades) > 0:
            trade_numbers = list(range(1, len(self.trades) + 1))
            cumulative_returns = np.cumsum([t.net_profit_loss for t in self.trades])
            cumulative_volume = np.cumsum([t.quantity * t.entry_price for t in self.trades])
            
            ax6_twin = ax6.twinx()
            ax6.plot(trade_numbers, cumulative_returns, label='Cumulative Returns', color='green', linewidth=2)
            ax6_twin.plot(trade_numbers, cumulative_volume, label='Cumulative Volume', color='blue', alpha=0.7)
            
            ax6.set_xlabel('Trade Number')
            ax6.set_ylabel('Cumulative Returns (USD)', color='green')
            ax6_twin.set_ylabel('Cumulative Volume (USD)', color='blue')
            ax6.set_title('Returns vs Volume')
            ax6.grid(True, alpha=0.3)
            
            # Add target line
            target_return = self.initial_balance * 0.5  # 50% target
            ax6.axhline(y=target_return, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('/home/f.kalati/Documents/crypto/ultra_aggressive_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """Main execution function for ultra-aggressive scalping."""
    print("🚀 ULTRA AGGRESSIVE DOGE/USD Scalping Bot v4.0")
    print("=" * 60)
    print("💀 EXTREME RISK - EXTREME REWARD")
    print("🎯 Target: 50%+ Monthly Returns")
    print("⚡ Leverage: 2x Simulated")
    print("🔥 MAXIMUM AGGRESSION MODE")
    print("💸 100+ Trades Target\n")
    
    print("🚨 EXTREME WARNING: This bot uses DANGEROUS parameters!")
    print("   💀 Up to 60% position sizes")
    print("   ⚡ 2x simulated leverage")
    print("   🔥 Trade every 30 seconds")
    print("   📈 Target 100+ trades per month")
    print("   Only use with money you can COMPLETELY AFFORD TO LOSE!\n")
    
    try:
        # Initialize ultra-aggressive backtesting engine
        backtest = UltraAggressiveBacktestingEngine(INITIAL_BALANCE)
        
        # Fetch data
        df = backtest.fetch_data()
        
        if df.empty:
            print("❌ No data available for backtesting")
            return
        
        print(f"📅 Data range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Total candles: {len(df):,}")
        
        # Run ultra-aggressive backtest
        performance = backtest.run_ultra_aggressive_backtest(df)
        
        if 'error' not in performance:
            # Print ultra-detailed results
            print("\n" + "=" * 80)
            print("🚀 ULTRA AGGRESSIVE PERFORMANCE REPORT")
            print("=" * 80)
            print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
            print(f"Final Balance: ${performance['final_balance']:,.2f}")
            print(f"Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.2%})")
            print(f"🎯 MONTHLY PROJECTION: {performance['monthly_return_projection']:.1%}")
            
            # TARGET ACHIEVEMENT CHECK
            target_achieved = performance['monthly_return_projection'] >= 50
            print(f"🎯 TARGET (50% monthly): {'🎉 ACHIEVED!' if target_achieved else '❌ NOT ACHIEVED'}")
            
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Win Rate: {performance['win_rate']:.1%}")
            print(f"Profit Factor: {performance['profit_factor']:.2f}")
            print(f"Average Leverage: {performance['avg_leverage']:.1f}x")
            print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
            
            # Ultra-aggressive specific metrics
            print(f"\n⚡ ULTRA-AGGRESSIVE METRICS")
            print("-" * 40)
            print(f"Total Volume Traded: ${performance['total_volume_traded']:,.0f}")
            print(f"Volume Turnover: {performance['volume_turnover']:.1f}x")
            print(f"Total Leverage Exposure: ${performance['total_leverage_exposure']:,.0f}")
            print(f"Average Holding Time: {performance['avg_holding_time_minutes']:.1f} minutes")
            
            # Fee analysis
            print(f"\n💸 FEE IMPACT ANALYSIS")
            print("-" * 40)
            print(f"Total Fees Paid: ${performance['total_fees_paid']:.2f}")
            print(f"Fee Impact: {performance['fee_impact_pct']:.2%} of capital")
            
            # Success analysis
            print(f"\n📊 SUCCESS ANALYSIS")
            print("-" * 40)
            if performance['total_trades'] >= 100:
                print("✅ EXCELLENT: 100+ trades achieved")
            elif performance['total_trades'] >= 50:
                print("✅ GOOD: 50+ trades achieved")
            else:
                print(f"⚠️ LOW: Only {performance['total_trades']} trades (target: 100+)")
            
            if target_achieved:
                print("🎉 SUCCESS: 50%+ monthly target ACHIEVED!")
                print("   💀 Ultra-aggressive strategy worked!")
                print("   ⚠️ Remember: This is EXTREME RISK trading")
            else:
                print(f"📉 PARTIAL: {performance['monthly_return_projection']:.1%} monthly (target: 50%+)")
                if performance['monthly_return_projection'] >= 25:
                    print("   💡 Good progress - consider more aggression")
                elif performance['monthly_return_projection'] >= 10:
                    print("   🔄 Moderate success - needs optimization")
                else:
                    print("   🚨 Poor performance - major adjustments needed")
            
            # Risk warnings
            print(f"\n🚨 RISK WARNINGS")
            print("-" * 40)
            if performance['max_drawdown'] > 0.15:
                print(f"⚠️ HIGH DRAWDOWN: {performance['max_drawdown']:.1%} (>15%)")
            if performance['fee_impact_pct'] > 0.05:
                print(f"⚠️ HIGH FEES: {performance['fee_impact_pct']:.1%} of capital")
            if performance['volume_turnover'] > 10:
                print(f"⚠️ EXTREME TURNOVER: {performance['volume_turnover']:.1f}x capital")
            
            # Generate ultra-aggressive visualization
            backtest.plot_ultra_aggressive_results(df, performance)
            
            print(f"\n📊 Ultra-aggressive results saved to: /home/f.kalati/Documents/crypto/ultra_aggressive_results.png")
            
            # Final ultra-aggressive assessment
            print(f"\n💀 FINAL ULTRA-AGGRESSIVE ASSESSMENT")
            print("=" * 60)
            if target_achieved and performance['total_trades'] >= 50:
                print("🎉 MISSION ACCOMPLISHED!")
                print("   🎯 50%+ monthly target ACHIEVED")
                print("   📈 High trading frequency SUCCESS")
                print("   💀 Ultra-aggressive strategy WORKS")
                print("   ⚠️ EXTREME RISK but EXTREME REWARD")
                print("   💡 Consider live implementation with SMALL amounts")
            elif performance['monthly_return_projection'] >= 25:
                print("📈 PARTIAL SUCCESS!")
                print("   💡 Good returns but below 50% target")
                print("   🔧 Consider even MORE aggressive parameters")
                print("   ⚡ Increase leverage or position sizes")
            else:
                print("🔄 NEEDS MORE AGGRESSION!")
                print("   📉 Returns below expectations")
                print("   💀 Try EVEN MORE extreme parameters")
                print("   ⚡ Consider 3x leverage or 80% position sizes")
            
        else:
            print(f"❌ Error in performance calculation: {performance['error']}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    main() 