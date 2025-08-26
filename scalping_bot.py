#!/usr/bin/env python3
"""
Advanced DOGE/USD Scalping Bot with Multi-Indicator Analysis
============================================================
A conservative scalping bot designed for minimal risk and consistent profits
using multiple technical indicators and strict risk management.

Based on Enhanced Crypto Volatility Analysis results showing DOGE/USD 
as the optimal scalping pair with high liquidity and good volatility.

Features:
- Multiple technical indicators for confirmation
- Conservative risk management
- Quick profit targets with tight stops
- Comprehensive backtesting engine
- Detailed performance analytics

Dependencies:
    pip install ccxt pandas numpy matplotlib seaborn ta-lib
    (Note: ta-lib requires separate installation - see README)

Usage:
    python scalping_bot.py
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Trading Configuration
SYMBOL = "DOGE/USD"
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
TIMEFRAME = "1m"  # 1-minute scalping
LOOKBACK_DAYS = 30  # Historical data for backtesting

# Risk Management Settings
INITIAL_BALANCE = 1000.0  # Starting capital in USD
POSITION_SIZE_PCT = 0.1   # Use 10% of balance per trade (conservative)
PROFIT_TARGET_PCT = 0.003 # 0.3% profit target (conservative scalping)
STOP_LOSS_PCT = 0.002     # 0.2% stop loss (tight risk control)
MAX_TRADES_PER_HOUR = 6   # Prevent overtrading

# Technical Indicator Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_DEVIATION = 2
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_MA_PERIOD = 20

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'buy' or 'sell'
    profit_loss: Optional[float]
    profit_pct: Optional[float]
    reason: str  # Entry/exit reason
    
class Signal(NamedTuple):
    """Trading signal with confidence level."""
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1 scale
    reason: str
    price: float

# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

class TechnicalIndicators:
    """Advanced technical indicators for scalping."""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, deviation: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * deviation),
            'middle': sma,
            'lower': sma - (std * deviation)
        }
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def volume_profile(volume: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators."""
        volume_ma = volume.rolling(window=period).mean()
        volume_ratio = volume / volume_ma
        
        return {
            'volume_ma': volume_ma,
            'volume_ratio': volume_ratio,
            'volume_spike': volume_ratio > 1.5  # Volume spike detection
        }
    
    @staticmethod
    def support_resistance(highs: pd.Series, lows: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate dynamic support and resistance levels."""
        resistance = highs.rolling(window=period).max()
        support = lows.rolling(window=period).min()
        
        return {
            'resistance': resistance,
            'support': support,
            'range': resistance - support
        }

# ---------------------------------------------------------------------------
# Scalping Strategy
# ---------------------------------------------------------------------------

class ScalpingStrategy:
    """Advanced multi-indicator scalping strategy."""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.last_trade_time = None
        self.trades_this_hour = 0
        self.current_hour = None
    
    def analyze_market(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Comprehensive market analysis using multiple indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Calculate all indicators
        analysis = {}
        
        # Trend indicators
        analysis['rsi'] = self.indicators.rsi(close, RSI_PERIOD)
        macd_data = self.indicators.macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        analysis.update(macd_data)
        
        # Volatility indicators
        bb_data = self.indicators.bollinger_bands(close, BB_PERIOD, BB_DEVIATION)
        analysis.update(bb_data)
        
        # Moving averages
        analysis['ema_fast'] = self.indicators.ema(close, EMA_FAST)
        analysis['ema_slow'] = self.indicators.ema(close, EMA_SLOW)
        
        # Volume analysis
        volume_data = self.indicators.volume_profile(volume, VOLUME_MA_PERIOD)
        analysis.update(volume_data)
        
        # Support/Resistance
        sr_data = self.indicators.support_resistance(high, low)
        analysis.update(sr_data)
        
        # Additional derived indicators
        analysis['bb_position'] = (close - analysis['lower']) / (analysis['upper'] - analysis['lower'])
        analysis['price_vs_ema_fast'] = close / analysis['ema_fast'] - 1
        analysis['ema_trend'] = analysis['ema_fast'] > analysis['ema_slow']
        
        return analysis
    
    def generate_signal(self, df: pd.DataFrame, analysis: Dict[str, pd.Series]) -> Signal:
        """Generate trading signal based on multiple confirmations."""
        if len(df) < max(RSI_PERIOD, MACD_SLOW, BB_PERIOD, EMA_SLOW):
            return Signal('hold', 0.0, 'Insufficient data', df['close'].iloc[-1])
        
        current_price = df['close'].iloc[-1]
        current_time = df.index[-1]
        
        # Check rate limiting
        if not self._can_trade(current_time):
            return Signal('hold', 0.0, 'Rate limited', current_price)
        
        # Get latest indicator values
        latest_idx = -1
        rsi = analysis['rsi'].iloc[latest_idx]
        macd = analysis['macd'].iloc[latest_idx]
        macd_signal = analysis['signal'].iloc[latest_idx]
        macd_hist = analysis['histogram'].iloc[latest_idx]
        bb_position = analysis['bb_position'].iloc[latest_idx]
        volume_ratio = analysis['volume_ratio'].iloc[latest_idx]
        ema_trend = analysis['ema_trend'].iloc[latest_idx]
        price_vs_ema = analysis['price_vs_ema_fast'].iloc[latest_idx]
        
        # Skip if any indicator is NaN
        if pd.isna([rsi, macd, macd_signal, bb_position, volume_ratio]).any():
            return Signal('hold', 0.0, 'Invalid indicator data', current_price)
        
        # Buy signal criteria (all must be true for high confidence)
        buy_signals = []
        buy_confidence = 0.0
        
        # RSI oversold but not extreme
        if RSI_OVERSOLD < rsi < 45:
            buy_signals.append("RSI recovering from oversold")
            buy_confidence += 0.2
        
        # MACD bullish momentum
        if macd > macd_signal and macd_hist > 0:
            buy_signals.append("MACD bullish crossover")
            buy_confidence += 0.25
        
        # Price near lower Bollinger Band (oversold)
        if 0.1 < bb_position < 0.3:
            buy_signals.append("Price near BB lower band")
            buy_confidence += 0.2
        
        # EMA trend confirmation
        if ema_trend and price_vs_ema > -0.002:  # Price close to or above fast EMA
            buy_signals.append("EMA trend bullish")
            buy_confidence += 0.2
        
        # Volume confirmation
        if volume_ratio > 1.2:  # Above average volume
            buy_signals.append("Volume confirmation")
            buy_confidence += 0.15
        
        # Conservative buy: require high confidence (multiple confirmations)
        if buy_confidence >= 0.7 and len(buy_signals) >= 4:
            reason = f"BUY: {', '.join(buy_signals)}"
            return Signal('buy', buy_confidence, reason, current_price)
        
        # Sell signal criteria (conservative exit)
        sell_signals = []
        sell_confidence = 0.0
        
        # RSI overbought
        if rsi > RSI_OVERBOUGHT:
            sell_signals.append("RSI overbought")
            sell_confidence += 0.3
        
        # MACD bearish momentum
        if macd < macd_signal and macd_hist < 0:
            sell_signals.append("MACD bearish crossover")
            sell_confidence += 0.25
        
        # Price near upper Bollinger Band (overbought)
        if bb_position > 0.8:
            sell_signals.append("Price near BB upper band")
            sell_confidence += 0.25
        
        # EMA trend weakening
        if not ema_trend or price_vs_ema < -0.001:
            sell_signals.append("EMA trend weakening")
            sell_confidence += 0.2
        
        # Conservative sell: require reasonable confidence
        if sell_confidence >= 0.5 and len(sell_signals) >= 2:
            reason = f"SELL: {', '.join(sell_signals)}"
            return Signal('sell', sell_confidence, reason, current_price)
        
        return Signal('hold', 0.0, 'No clear signal', current_price)
    
    def _can_trade(self, current_time: datetime) -> bool:
        """Check if we can trade based on rate limits."""
        current_hour = current_time.hour
        
        # Reset hourly counter
        if self.current_hour != current_hour:
            self.current_hour = current_hour
            self.trades_this_hour = 0
        
        # Check if we've exceeded hourly limit
        if self.trades_this_hour >= MAX_TRADES_PER_HOUR:
            return False
        
        # Check minimum time between trades (prevent overtrading)
        if self.last_trade_time:
            time_diff = (current_time - self.last_trade_time).total_seconds()
            if time_diff < 300:  # Minimum 5 minutes between trades
                return False
        
        return True
    
    def update_trade_time(self, trade_time: datetime):
        """Update the last trade time and counter."""
        self.last_trade_time = trade_time
        self.trades_this_hour += 1

# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------

class BacktestingEngine:
    """Comprehensive backtesting engine for scalping strategies."""
    
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0  # Current position size
        self.trades: List[Trade] = []
        self.strategy = ScalpingStrategy()
        self.open_trade: Optional[Trade] = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical 1-minute data for DOGE/USD."""
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
                since_ms = ohlcv[-1][0] + 60000  # Move to next minute
                time.sleep(0.1)  # Rate limiting
                
                # Progress indicator
                progress = (since_ms - int(start_time.timestamp() * 1000)) / (end_ms - int(start_time.timestamp() * 1000))
                print(f"\r   Progress: {progress:.1%}", end="", flush=True)
                
            except Exception as e:
                print(f"\n   ⚠️ Error fetching data: {e}")
                break
        
        print(f"\n   ✅ Fetched {len(all_data)} candles")
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Clean data
        df = df.astype(float)
        df = df.dropna()
        
        return df
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run comprehensive backtest on historical data."""
        print(f"\n🚀 Starting backtest with ${self.initial_balance:,.2f} initial balance")
        print(f"📈 Strategy: Conservative scalping with {PROFIT_TARGET_PCT:.1%} profit target")
        print(f"🛡️ Risk: {STOP_LOSS_PCT:.1%} stop loss, {POSITION_SIZE_PCT:.1%} position size")
        print("=" * 60)
        
        # Calculate indicators for entire dataset
        analysis = self.strategy.analyze_market(df)
        
        # Add analysis to dataframe for easier access
        for key, series in analysis.items():
            df[f'ind_{key}'] = series
        
        total_signals = 0
        buy_signals = 0
        
        # Iterate through each minute
        for i in range(max(RSI_PERIOD, MACD_SLOW, BB_PERIOD), len(df)):
            current_time = df.index[i]
            current_data = df.iloc[:i+1]  # Data up to current point
            current_analysis = {key: series.iloc[:i+1] for key, series in analysis.items()}
            
            # Generate signal
            signal = self.strategy.generate_signal(current_data, current_analysis)
            total_signals += 1
            
            if signal.action != 'hold':
                print(f"🔔 {current_time.strftime('%Y-%m-%d %H:%M')} - {signal.action.upper()}: {signal.reason}")
            
            # Process signal
            if signal.action == 'buy' and self.open_trade is None:
                self._execute_buy(current_time, signal.price, signal.reason)
                buy_signals += 1
                
            elif signal.action == 'sell' and self.open_trade is not None:
                self._execute_sell(current_time, signal.price, signal.reason)
            
            # Check stop loss and profit target for open trades
            if self.open_trade is not None:
                self._check_exit_conditions(current_time, df.iloc[i])
        
        # Close any remaining open position
        if self.open_trade is not None:
            final_price = df['close'].iloc[-1]
            final_time = df.index[-1]
            self._execute_sell(final_time, final_price, "End of backtest")
        
        # Calculate performance metrics
        performance = self._calculate_performance(df)
        
        print(f"\n📊 Backtest completed: {total_signals:,} signals analyzed, {buy_signals} buy signals")
        
        return performance
    
    def _execute_buy(self, timestamp: datetime, price: float, reason: str):
        """Execute buy order."""
        if self.balance <= 0:
            return
        
        trade_amount = self.balance * POSITION_SIZE_PCT
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
            reason=reason
        )
        
        self.balance -= trade_amount
        self.position = quantity
        self.strategy.update_trade_time(timestamp)
        
        print(f"  💰 BUY: {quantity:.2f} DOGE at ${price:.6f} (${trade_amount:.2f})")
    
    def _execute_sell(self, timestamp: datetime, price: float, reason: str):
        """Execute sell order."""
        if self.open_trade is None:
            return
        
        trade_value = self.position * price
        profit_loss = trade_value - (self.open_trade.quantity * self.open_trade.entry_price)
        profit_pct = profit_loss / (self.open_trade.quantity * self.open_trade.entry_price)
        
        # Complete the trade
        self.open_trade.exit_time = timestamp
        self.open_trade.exit_price = price
        self.open_trade.profit_loss = profit_loss
        self.open_trade.profit_pct = profit_pct
        
        self.balance += trade_value
        self.trades.append(self.open_trade)
        
        status = "✅ PROFIT" if profit_loss > 0 else "❌ LOSS"
        print(f"  🔄 SELL: {self.position:.2f} DOGE at ${price:.6f} (${trade_value:.2f}) - {status}: ${profit_loss:.2f} ({profit_pct:.2%})")
        
        self.position = 0.0
        self.open_trade = None
        self.strategy.update_trade_time(timestamp)
    
    def _check_exit_conditions(self, timestamp: datetime, candle: pd.Series):
        """Check stop loss and profit target conditions."""
        if self.open_trade is None:
            return
        
        current_price = candle['close']
        entry_price = self.open_trade.entry_price
        
        # Calculate current profit/loss percentage
        current_pnl_pct = (current_price - entry_price) / entry_price
        
        # Profit target reached
        if current_pnl_pct >= PROFIT_TARGET_PCT:
            self._execute_sell(timestamp, current_price, f"Profit target reached ({current_pnl_pct:.2%})")
            return
        
        # Stop loss triggered
        if current_pnl_pct <= -STOP_LOSS_PCT:
            self._execute_sell(timestamp, current_price, f"Stop loss triggered ({current_pnl_pct:.2%})")
            return
    
    def _calculate_performance(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'profit_loss': t.profit_loss,
            'profit_pct': t.profit_pct,
            'reason': t.reason
        } for t in self.trades])
        
        # Basic metrics
        total_return = self.balance - self.initial_balance
        total_return_pct = total_return / self.initial_balance
        
        winning_trades = trades_df[trades_df['profit_loss'] > 0]
        losing_trades = trades_df[trades_df['profit_loss'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0
        
        # Risk metrics
        profit_factor = abs(winning_trades['profit_loss'].sum() / losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 and losing_trades['profit_loss'].sum() != 0 else float('inf')
        
        # Timing metrics
        holding_times = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in self.trades if t.exit_time]
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
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
            'trades_data': trades_df
        }
    
    def plot_results(self, df: pd.DataFrame, performance: Dict):
        """Create comprehensive performance visualization."""
        if 'error' in performance:
            print(f"❌ Cannot plot: {performance['error']}")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'DOGE/USD Scalping Bot Performance - {len(self.trades)} Trades', fontsize=16, fontweight='bold')
        
        # 1. Price chart with trades
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='DOGE/USD Price', alpha=0.7, linewidth=1)
        
        # Mark trades
        for trade in self.trades[:50]:  # Show first 50 trades to avoid clutter
            color = 'green' if trade.profit_loss > 0 else 'red'
            ax1.scatter(trade.entry_time, trade.entry_price, color='blue', marker='^', s=50, alpha=0.7)
            if trade.exit_time:
                ax1.scatter(trade.exit_time, trade.exit_price, color=color, marker='v', s=50, alpha=0.7)
        
        ax1.set_title('Price Chart with Trade Markers')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Equity curve
        ax2 = axes[0, 1]
        equity_curve = [self.initial_balance]
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade.profit_loss)
        
        ax2.plot(range(len(equity_curve)), equity_curve, color='green', linewidth=2)
        ax2.axhline(y=self.initial_balance, color='red', linestyle='--', alpha=0.7, label='Starting Balance')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Balance (USD)')
        ax2.set_xlabel('Trade Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade distribution
        ax3 = axes[1, 0]
        profits = [t.profit_loss for t in self.trades]
        ax3.hist(profits, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Trade P&L Distribution')
        ax3.set_xlabel('Profit/Loss (USD)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Win/Loss pie chart
        ax4 = axes[1, 1]
        sizes = [performance['winning_trades'], performance['losing_trades']]
        labels = [f"Wins ({performance['win_rate']:.1%})", f"Losses ({1-performance['win_rate']:.1%})"]
        colors = ['lightgreen', 'lightcoral']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
        ax4.set_title('Win/Loss Ratio')
        
        # 5. Performance metrics table
        ax5 = axes[2, 0]
        ax5.axis('off')
        metrics_text = f"""
        📊 PERFORMANCE SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Trades: {performance['total_trades']}
        Win Rate: {performance['win_rate']:.1%}
        Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.1%})
        Average Win: ${performance['avg_win']:.2f}
        Average Loss: ${performance['avg_loss']:.2f}
        Profit Factor: {performance['profit_factor']:.2f}
        Avg Holding Time: {performance['avg_holding_time_minutes']:.1f} minutes
        Final Balance: ${performance['final_balance']:,.2f}
        """
        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace')
        
        # 6. Monthly returns
        ax6 = axes[2, 1]
        if len(self.trades) > 0:
            trades_df = performance['trades_data']
            trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
            monthly_pnl = trades_df.groupby('month')['profit_loss'].sum()
            
            bars = ax6.bar(range(len(monthly_pnl)), monthly_pnl.values, 
                          color=['green' if x > 0 else 'red' for x in monthly_pnl.values])
            ax6.set_title('Monthly P&L')
            ax6.set_ylabel('Profit/Loss (USD)')
            ax6.set_xlabel('Month')
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)
            
            # Add month labels
            month_labels = [str(m) for m in monthly_pnl.index]
            ax6.set_xticks(range(len(monthly_pnl)))
            ax6.set_xticklabels(month_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/f.kalati/Documents/crypto/scalping_results.png', dpi=300, bbox_inches='tight')
        plt.show()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """Main execution function."""
    print("🚀 DOGE/USD Advanced Scalping Bot")
    print("==================================")
    print("Based on Enhanced Crypto Volatility Analysis")
    print("Conservative scalping with multi-indicator confirmation\n")
    
    try:
        # Initialize backtesting engine
        backtest = BacktestingEngine(INITIAL_BALANCE)
        
        # Fetch historical data
        df = backtest.fetch_data()
        
        if df.empty:
            print("❌ No data available for backtesting")
            return
        
        print(f"📅 Data range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Total candles: {len(df):,}")
        
        # Run backtest
        performance = backtest.run_backtest(df)
        
        if 'error' not in performance:
            # Print detailed results
            print("\n" + "=" * 80)
            print("📈 FINAL PERFORMANCE REPORT")
            print("=" * 80)
            print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
            print(f"Final Balance: ${performance['final_balance']:,.2f}")
            print(f"Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.2%})")
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Win Rate: {performance['win_rate']:.1%}")
            print(f"Winning Trades: {performance['winning_trades']}")
            print(f"Losing Trades: {performance['losing_trades']}")
            print(f"Average Win: ${performance['avg_win']:.2f}")
            print(f"Average Loss: ${performance['avg_loss']:.2f}")
            print(f"Profit Factor: {performance['profit_factor']:.2f}")
            print(f"Average Holding Time: {performance['avg_holding_time_minutes']:.1f} minutes")
            
            # Risk assessment
            print(f"\n🛡️ RISK ASSESSMENT")
            print("-" * 40)
            if performance['win_rate'] >= 0.6:
                print("✅ Excellent win rate (>60%)")
            elif performance['win_rate'] >= 0.5:
                print("✅ Good win rate (>50%)")
            else:
                print("⚠️ Low win rate (<50%) - Review strategy")
            
            if performance['profit_factor'] >= 2.0:
                print("✅ Excellent profit factor (>2.0)")
            elif performance['profit_factor'] >= 1.5:
                print("✅ Good profit factor (>1.5)")
            else:
                print("⚠️ Low profit factor (<1.5) - Review risk management")
            
            if performance['avg_holding_time_minutes'] <= 30:
                print("✅ Fast scalping strategy (≤30 min avg hold)")
            else:
                print("ℹ️ Longer holding times than typical scalping")
            
            # Generate visualization
            backtest.plot_results(df, performance)
            
            print(f"\n📊 Results saved to: /home/f.kalati/Documents/crypto/scalping_results.png")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    main() 