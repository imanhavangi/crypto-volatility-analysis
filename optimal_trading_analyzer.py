#!/usr/bin/env python3
"""
Optimal Trading Point Analyzer for DOGE/USD
==========================================
This script finds the PERFECT entry and exit points using hindsight
to establish the maximum possible profit as a benchmark for AI training.

Phase 1: Perfect Hindsight Analysis
- Fetch 1-month DOGE/USD 1-minute data
- Find optimal buy/sell points
- Calculate maximum possible profit
- Generate training labels for AI

Phase 2: Deep Learning Model (Next step)
- Train neural network to predict optimal points
- Use candlestick patterns and indicators as features
- Target: Get as close as possible to optimal profit

Dependencies:
    pip install ccxt pandas numpy matplotlib seaborn scikit-learn tensorflow

Usage:
    python optimal_trading_analyzer.py
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

SYMBOL = "DOGE/USDT"  # Binance uses USDT
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
TIMEFRAME = "1m"
LOOKBACK_DAYS = 30  # Target: Full month

# Trading constraints for realistic analysis
INITIAL_BALANCE = 1000.0
TRADING_FEE = 0.0026  # 0.26% taker fee (realistic)
MIN_TRADE_AMOUNT = 10.0  # Minimum $10 per trade
MAX_POSITION_SIZE_PCT = 1.0  # 100% of balance (all-in for max profit)

# Optimal trading parameters - Aggressive settings for maximum trades
MIN_PROFIT_PCT = 0.0005  # Minimum 0.05% profit to consider a trade (very aggressive)
MIN_HOLD_TIME_MINUTES = 1  # Minimum holding time
MAX_HOLD_TIME_MINUTES = 120  # Maximum holding time for scalping

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class OptimalTrade:
    """Represents a perfect trade with hindsight."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    profit_pct: float
    gross_profit: float
    net_profit: float
    fees_paid: float
    hold_time_minutes: float
    reason: str

@dataclass
class MarketCondition:
    """Market condition at a given time for feature engineering."""
    timestamp: datetime
    price: float
    volume: float
    volatility_5m: float
    volatility_15m: float
    rsi_14: float
    macd: float
    macd_signal: float
    bb_position: float  # Position within Bollinger Bands (0-1)
    volume_ratio: float  # Volume vs average
    price_change_1m: float
    price_change_5m: float
    price_change_15m: float
    is_optimal_entry: bool  # Label for AI training
    is_optimal_exit: bool   # Label for AI training

# ---------------------------------------------------------------------------
# Data Fetching and Processing
# ---------------------------------------------------------------------------

class OptimalTradingAnalyzer:
    """Analyzer to find optimal trading points with perfect hindsight."""
    
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.optimal_trades: List[OptimalTrade] = []
        self.market_conditions: List[MarketCondition] = []
        
    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch comprehensive historical data for analysis."""
        print(f"📊 Fetching {LOOKBACK_DAYS} days (full month) of 1-minute data...")
        
        # Try multiple exchanges for better data coverage
        exchanges_to_try = [
            ('binance', {
                'enableRateLimit': True,
                'rateLimit': 1200,  # Binance rate limit
                'timeout': 30000,
                'options': {'adjustForTimeDifference': True}
            }),
            ('okx', {
                'enableRateLimit': True,
                'rateLimit': 2000,
                'timeout': 30000,
            }),
            ('kraken', {
                'enableRateLimit': True,
                'rateLimit': 3000,
                'timeout': 60000,
            })
        ]
        
        exchange = None
        symbol_to_use = SYMBOL
        
        for exchange_name, config in exchanges_to_try:
            try:
                print(f"   🔗 Trying {exchange_name.upper()}...")
                
                # Adjust symbol for different exchanges
                if exchange_name == 'binance':
                    symbol_to_use = "DOGE/USDT"
                elif exchange_name == 'okx':
                    symbol_to_use = "DOGE/USDT"  
                elif exchange_name == 'kraken':
                    symbol_to_use = "DOGE/USD"
                
                exchange = getattr(ccxt, exchange_name)(config)
                exchange.load_markets()
                
                # Test if symbol exists
                if symbol_to_use in exchange.markets:
                    print(f"   ✅ Connected to {exchange_name.upper()} with {symbol_to_use}")
                    break
                else:
                    print(f"   ❌ {symbol_to_use} not available on {exchange_name.upper()}")
                    exchange = None
                    
            except Exception as e:
                print(f"   ❌ Failed to connect to {exchange_name.upper()}: {e}")
                exchange = None
                continue
        
        if not exchange:
            raise Exception("Could not connect to any exchange!")
        
        print(f"   🎯 Using {exchange.id.upper()} for {symbol_to_use}")
        
        # Try to get more data by starting earlier
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=LOOKBACK_DAYS)
        
        print(f"   📅 Requesting data from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        
        all_data = []
        target_candles = LOOKBACK_DAYS * 24 * 60  # Total 1-minute candles for full period
        
        # Optimize batch size based on exchange
        if exchange.id.lower() == 'binance':
            batch_size = 1000  # Binance supports up to 1000
        elif exchange.id.lower() == 'okx':
            batch_size = 1000  # OKX supports up to 1000
        else:
            batch_size = 720   # Conservative for others
        
        print(f"   🎯 Target: {target_candles:,} candles ({LOOKBACK_DAYS} days)")
        print(f"   📦 Batch size: {batch_size} candles per request")
        
        # Start from current time and go backwards
        current_end_ms = int(end_time.timestamp() * 1000)
        
        batch_count = 0
        max_batches = (target_candles // batch_size) + 5  # Add buffer for safety
        ohlcv = []  # Initialize to avoid scope issues
        consecutive_empty_batches = 0  # Track batches with no new data
        
        while len(all_data) < target_candles and batch_count < max_batches:
            batch_count += 1
            retry_count = 0
            max_retries = 3
            batch_success = False
            
            while not batch_success and retry_count < max_retries:
                try:
                    # Calculate time window for this batch (720 candles = 12 hours)
                    batch_start_ms = current_end_ms - (batch_size * 60 * 1000)
                    
                    ohlcv = exchange.fetch_ohlcv(
                        symbol_to_use, 
                        TIMEFRAME, 
                        since=batch_start_ms, 
                        limit=batch_size
                    )
                    
                    if not ohlcv or len(ohlcv) < 50:  # If less than 50 candles, probably reached the end
                        print(f"\n   ⚠️ Insufficient data in batch {batch_count} ({len(ohlcv) if ohlcv else 0} candles) - stopping")
                        batch_success = True  # Exit the retry loop
                        break
                    
                    # Remove duplicates based on timestamp
                    unique_ohlcv = []
                    seen_timestamps = set()
                    for candle in ohlcv:
                        if candle[0] not in seen_timestamps:
                            unique_ohlcv.append(candle)
                            seen_timestamps.add(candle[0])
                    
                    if len(unique_ohlcv) < len(ohlcv):
                        print(f"\n   🔄 Removed {len(ohlcv) - len(unique_ohlcv)} duplicate candles")
                    
                    ohlcv = unique_ohlcv
                    
                    # Debug info
                    first_candle_time = datetime.fromtimestamp(ohlcv[0][0]/1000, timezone.utc)
                    last_candle_time = datetime.fromtimestamp(ohlcv[-1][0]/1000, timezone.utc)
                    
                    # Add to beginning of list (since we're going backwards)
                    # But only add candles that we don't already have
                    new_candles = []
                    existing_timestamps = {candle[0] for candle in all_data} if all_data else set()
                    
                    for candle in ohlcv:
                        if candle[0] not in existing_timestamps:
                            new_candles.append(candle)
                    
                    all_data = new_candles + all_data
                    
                    print(f"\n   ✅ Batch {batch_count}: {len(ohlcv)} candles ({len(new_candles)} new) from {first_candle_time.strftime('%Y-%m-%d %H:%M')} to {last_candle_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   📈 Total collected: {len(all_data):,} candles")
                    
                    progress = min(len(all_data) / target_candles, 1.0)
                    print(f"   🎯 Progress: {progress:.1%}")
                    
                    # Move end time backwards for next batch (only on success)
                    # If we got no new candles, we need to go back further
                    if len(new_candles) == 0:
                        consecutive_empty_batches += 1
                        print(f"   ⚠️ No new candles in this batch ({consecutive_empty_batches} consecutive empty) - going back further")
                        current_end_ms = ohlcv[0][0] - (batch_size * 60 * 1000)  # Go back full batch size
                        
                        # Stop if too many consecutive empty batches
                        if consecutive_empty_batches >= 3:
                            print(f"   🛑 Stopping after {consecutive_empty_batches} consecutive empty batches")
                            batch_success = True  # Exit inner loop
                            break
                    else:
                        consecutive_empty_batches = 0  # Reset counter
                        current_end_ms = ohlcv[0][0] - 60000  # Start 1 minute before first candle
                    
                    batch_success = True
                    
                    # Exchange-specific delays
                    if exchange.id.lower() == 'binance':
                        time.sleep(0.5)  # Binance is fast
                    elif exchange.id.lower() == 'okx':
                        time.sleep(0.8)  # OKX moderate
                    else:
                        time.sleep(1.5)  # Conservative for others
                    
                    # Stop if we've collected enough data
                    if len(all_data) >= target_candles:
                        print(f"\n   ✅ Target reached! Collected {len(all_data):,} candles")
                        break
                        
                except Exception as e:
                    retry_count += 1
                    print(f"\n   ⚠️ Error in batch {batch_count}, attempt {retry_count}/{max_retries}: {e}")
                    
                    if retry_count < max_retries:
                        # Exchange-specific error delays
                        if exchange.id.lower() == 'binance':
                            wait_time = 3 * retry_count  # 3s, 6s, 9s
                        elif exchange.id.lower() == 'okx':
                            wait_time = 4 * retry_count  # 4s, 8s, 12s
                        else:
                            wait_time = 5 * retry_count  # 5s, 10s, 15s
                        
                        print(f"   ⏳ Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ Failed to fetch batch {batch_count} after {max_retries} attempts, skipping...")
                        batch_success = True  # Exit retry loop to continue with next batch
            
            # Check if we should break out of outer loop
            if not ohlcv or len(ohlcv) < 50 or consecutive_empty_batches >= 3:
                if consecutive_empty_batches >= 3:
                    print(f"   🏁 Data collection stopped due to consecutive empty batches")
                break
        
        print(f"\n   ✅ Fetched {len(all_data):,} candles")
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Remove duplicate timestamps to avoid index errors
        print(f"   🔍 Checking for duplicate timestamps...")
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"   🧹 Removed {initial_count - final_count} duplicate timestamps")
        
        df.set_index('timestamp', inplace=True)
        df = df.astype(float).dropna()
        
        # Sort by timestamp to ensure proper order
        df = df.sort_index()
        
        print(f"   ✅ Final dataset: {len(df):,} unique candles")
        print(f"   📅 Date range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators ONLY for feature storage - NOT used in optimal trading algorithm."""
        print("🔧 Calculating technical indicators for feature engineering (not used in trading logic)...")
        
        # Price-based indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_15'] = df['close'].rolling(15).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std_dev = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility measures
        df['volatility_5m'] = df['close'].rolling(5).std()
        df['volatility_15m'] = df['close'].rolling(15).std()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price changes
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_5m'] = df['close'].pct_change(5)
        df['price_change_15m'] = df['close'].pct_change(15)
        
        # Clean data
        df = df.dropna()
        
        print(f"   ✅ Calculated indicators for {len(df):,} periods")
        return df
    
    def find_optimal_entry_exit_points(self, df: pd.DataFrame) -> List[OptimalTrade]:
        """Find PERFECT entry and exit points using complete hindsight analysis.
        
        Algorithm:
        1. Find all local minima (valleys) for optimal entry points
        2. Find all local maxima (peaks) for optimal exit points  
        3. Match entries with best possible exits
        4. Maximize number of trades and total profit
        """
        print("🎯 Finding optimal entry/exit points with PERFECT hindsight...")
        print("   📊 Using complete price data to find exact tops and bottoms")
        
        # Get price data
        prices = df['close'].values
        timestamps = df.index.values
        
        optimal_trades = []
        current_balance = self.initial_balance
        
        # Find local minima (entry points) and maxima (exit points)
        print("   🔍 Detecting local extremes...")
        
        # Use a rolling window to find local extremes
        window_size = 2  # Look 2 minutes in each direction for maximum opportunities
        local_minima = []  # (index, price, timestamp)
        local_maxima = []  # (index, price, timestamp)
        
        for i in range(window_size, len(prices) - window_size):
            current_price = prices[i]
            
            # Check if this is a local minimum
            is_local_min = True
            is_local_max = True
            
            # Compare with surrounding prices
            for j in range(i - window_size, i + window_size + 1):
                if j != i:
                    if prices[j] <= current_price:
                        is_local_min = False
                    if prices[j] >= current_price:
                        is_local_max = False
            
            if is_local_min:
                local_minima.append((i, current_price, timestamps[i]))
            
            if is_local_max:
                local_maxima.append((i, current_price, timestamps[i]))
        
        print(f"   ✅ Found {len(local_minima)} local minima (entry candidates)")
        print(f"   ✅ Found {len(local_maxima)} local maxima (exit candidates)")
        
        # Now match each minimum with the best maximum that comes after it
        print("   🔄 Matching optimal entry/exit pairs...")
        
        used_maxima = set()  # Track used exit points
        
        for min_idx, entry_price, entry_time in local_minima:
            best_exit = None
            best_profit_pct = 0
            
            # Find the best maximum that comes after this minimum
            for max_idx, exit_price, exit_time in local_maxima:
                # Must come after entry and not be used
                if max_idx > min_idx and max_idx not in used_maxima:
                    # Calculate gross profit percentage
                    gross_profit_pct = (exit_price - entry_price) / entry_price
                    
                    # Account for fees
                    net_profit_pct = gross_profit_pct - (2 * TRADING_FEE)
                    
                    # Check if profitable and better than current best
                    if net_profit_pct > MIN_PROFIT_PCT and net_profit_pct > best_profit_pct:
                        # Make sure no other minimum between entry and exit would be better
                        is_valid_exit = True
                        for other_min_idx, other_price, _ in local_minima:
                            if min_idx < other_min_idx < max_idx and other_price < entry_price:
                                is_valid_exit = False
                                break
                        
                        if is_valid_exit:
                            best_exit = (max_idx, exit_price, exit_time)
                            best_profit_pct = net_profit_pct
            
            # If we found a good exit, create the trade
            if best_exit is not None:
                exit_idx, exit_price, exit_time = best_exit
                used_maxima.add(exit_idx)
                
                # Calculate trade details
                trade_amount = current_balance * MAX_POSITION_SIZE_PCT
                quantity = trade_amount / entry_price
                
                gross_profit = quantity * (exit_price - entry_price)
                entry_fee = trade_amount * TRADING_FEE
                exit_fee = (quantity * exit_price) * TRADING_FEE
                total_fees = entry_fee + exit_fee
                net_profit = gross_profit - total_fees
                
                hold_time = (pd.to_datetime(exit_time) - pd.to_datetime(entry_time)).total_seconds() / 60
                
                # Create optimal trade
                trade = OptimalTrade(
                    entry_time=pd.to_datetime(entry_time),
                    exit_time=pd.to_datetime(exit_time),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    profit_pct=best_profit_pct,
                    gross_profit=gross_profit,
                    net_profit=net_profit,
                    fees_paid=total_fees,
                    hold_time_minutes=hold_time,
                    reason=f"Perfect hindsight: {best_profit_pct:.3%} profit"
                )
                
                optimal_trades.append(trade)
                current_balance += net_profit
                
                if len(optimal_trades) % 50 == 0:
                    print(f"\r   Found {len(optimal_trades)} optimal trades, Balance: ${current_balance:,.2f}...", end="", flush=True)
        
        # Sort trades by entry time
        optimal_trades.sort(key=lambda t: t.entry_time)
        
        print(f"\n   ✅ Found {len(optimal_trades)} PERFECT trades using complete hindsight")
        print(f"   💰 Perfect trading balance: ${current_balance:,.2f}")
        print(f"   📈 Perfect return: {((current_balance - self.initial_balance) / self.initial_balance):.1%}")
        
        # Additional analysis for very profitable short-term opportunities
        print("   🔍 Finding additional short-term opportunities...")
        
        # Look for smaller but profitable moves between the main trades
        additional_trades = self._find_additional_opportunities(df, optimal_trades)
        if additional_trades:
            optimal_trades.extend(additional_trades)
            optimal_trades.sort(key=lambda t: t.entry_time)
            
            # Recalculate final balance
            final_balance = self.initial_balance
            for trade in optimal_trades:
                final_balance += trade.net_profit
            
            print(f"   ✅ Total trades with micro-opportunities: {len(optimal_trades)}")
            print(f"   💰 Enhanced perfect balance: ${final_balance:,.2f}")
        
        return optimal_trades
    
    def _find_additional_opportunities(self, df: pd.DataFrame, main_trades: List[OptimalTrade]) -> List[OptimalTrade]:
        """Find additional profitable opportunities between main trades."""
        additional_trades = []
        
        # Create a set of time ranges already used by main trades
        used_ranges = []
        for trade in main_trades:
            used_ranges.append((trade.entry_time, trade.exit_time))
        
        # Look for opportunities in unused time periods
        prices = df['close'].values
        timestamps = df.index.values
        
        # Use smaller windows for micro-opportunities
        window_size = 2  # 2-minute windows
        min_profit_threshold = 0.001  # 0.1% minimum profit for micro trades
        
        for i in range(window_size, len(prices) - window_size, 3):  # Skip every 3 minutes to avoid overlap
            current_time = pd.to_datetime(timestamps[i])
            
            # Check if this time is already used
            is_time_used = False
            for start_time, end_time in used_ranges:
                if start_time <= current_time <= end_time:
                    is_time_used = True
                    break
            
            if is_time_used:
                continue
            
            current_price = prices[i]
            
            # Look for quick profit opportunities in next 10-30 minutes
            for j in range(i + 5, min(i + 30, len(prices))):  # 5-30 minutes ahead
                future_price = prices[j]
                future_time = pd.to_datetime(timestamps[j])
                
                # Check if future time conflicts with existing trades
                conflicts = False
                for start_time, end_time in used_ranges:
                    if start_time <= future_time <= end_time:
                        conflicts = True
                        break
                
                if conflicts:
                    continue
                
                # Calculate profit
                gross_profit_pct = (future_price - current_price) / current_price
                net_profit_pct = gross_profit_pct - (2 * TRADING_FEE)
                
                if net_profit_pct > min_profit_threshold:
                    # This is a micro opportunity
                    trade_amount = self.initial_balance * 0.5  # Use 50% for micro trades
                    quantity = trade_amount / current_price
                    
                    gross_profit = quantity * (future_price - current_price)
                    entry_fee = trade_amount * TRADING_FEE
                    exit_fee = (quantity * future_price) * TRADING_FEE
                    total_fees = entry_fee + exit_fee
                    net_profit = gross_profit - total_fees
                    
                    if net_profit > 1:  # At least $1 profit
                        hold_time = (future_time - current_time).total_seconds() / 60
                        
                        trade = OptimalTrade(
                            entry_time=current_time,
                            exit_time=future_time,
                            entry_price=current_price,
                            exit_price=future_price,
                            profit_pct=net_profit_pct,
                            gross_profit=gross_profit,
                            net_profit=net_profit,
                            fees_paid=total_fees,
                            hold_time_minutes=hold_time,
                            reason=f"Micro opportunity: {net_profit_pct:.3%}"
                        )
                        
                        additional_trades.append(trade)
                        used_ranges.append((current_time, future_time))
                        break  # Found opportunity for this entry point
        
        print(f"   ✅ Found {len(additional_trades)} additional micro-opportunities")
        return additional_trades
    
    def create_training_labels(self, df: pd.DataFrame, optimal_trades: List[OptimalTrade]) -> pd.DataFrame:
        """Create training labels for AI model based on optimal trades."""
        print("🏷️ Creating training labels for AI model...")
        
        # Initialize labels
        df['is_optimal_entry'] = False
        df['is_optimal_exit'] = False
        df['future_profit_potential'] = 0.0
        
        # Mark optimal entry and exit points
        for trade in optimal_trades:
            # Find closest indices - ensure timezone compatibility
            entry_time_utc = trade.entry_time.tz_convert('UTC') if trade.entry_time.tz is not None else trade.entry_time.tz_localize('UTC')
            exit_time_utc = trade.exit_time.tz_convert('UTC') if trade.exit_time.tz is not None else trade.exit_time.tz_localize('UTC')
            
            entry_idx = df.index.get_indexer([entry_time_utc], method='nearest')[0]
            exit_idx = df.index.get_indexer([exit_time_utc], method='nearest')[0]
            
            if entry_idx >= 0 and exit_idx >= 0:
                df.iloc[entry_idx, df.columns.get_loc('is_optimal_entry')] = True
                df.iloc[exit_idx, df.columns.get_loc('is_optimal_exit')] = True
                df.iloc[entry_idx, df.columns.get_loc('future_profit_potential')] = trade.profit_pct
        
        # Calculate additional features for AI
        df['trend_strength'] = df['close'].rolling(10).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
        )
        
        df['volatility_percentile'] = df['volatility_15m'].rolling(100).rank(pct=True)
        df['volume_percentile'] = df['volume'].rolling(100).rank(pct=True)
        
        print(f"   ✅ Created labels for {len(df)} data points")
        print(f"   📊 Entry signals: {df['is_optimal_entry'].sum()}")
        print(f"   📊 Exit signals: {df['is_optimal_exit'].sum()}")
        
        return df
    
    def analyze_optimal_performance(self, optimal_trades: List[OptimalTrade]) -> Dict:
        """Analyze the performance of optimal trading strategy."""
        if not optimal_trades:
            return {'error': 'No optimal trades found'}
        
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'profit_pct': t.profit_pct,
            'gross_profit': t.gross_profit,
            'net_profit': t.net_profit,
            'fees_paid': t.fees_paid,
            'hold_time_minutes': t.hold_time_minutes,
            'reason': t.reason
        } for t in optimal_trades])
        
        # Calculate performance metrics
        total_profit = trades_df['net_profit'].sum()
        total_return = total_profit / self.initial_balance
        final_balance = self.initial_balance + total_profit
        
        # Time-based metrics
        first_trade = trades_df['entry_time'].min()
        last_trade = trades_df['exit_time'].max()
        total_days = (last_trade - first_trade).days
        monthly_return = (total_return * 30 / total_days) if total_days > 0 else 0
        
        # Trade statistics
        avg_profit_pct = trades_df['profit_pct'].mean()
        median_profit_pct = trades_df['profit_pct'].median()
        max_profit_pct = trades_df['profit_pct'].max()
        min_profit_pct = trades_df['profit_pct'].min()
        
        # Timing statistics
        avg_hold_time = trades_df['hold_time_minutes'].mean()
        median_hold_time = trades_df['hold_time_minutes'].median()
        
        # Fee analysis
        total_fees = trades_df['fees_paid'].sum()
        fee_impact = total_fees / self.initial_balance
        
        return {
            'total_trades': len(optimal_trades),
            'total_profit': total_profit,
            'total_return_pct': total_return,
            'final_balance': final_balance,
            'monthly_return_projection': monthly_return,
            'avg_profit_pct': avg_profit_pct,
            'median_profit_pct': median_profit_pct,
            'max_profit_pct': max_profit_pct,
            'min_profit_pct': min_profit_pct,
            'avg_hold_time_minutes': avg_hold_time,
            'median_hold_time_minutes': median_hold_time,
            'total_fees_paid': total_fees,
            'fee_impact_pct': fee_impact,
            'trades_per_day': len(optimal_trades) / total_days if total_days > 0 else 0,
            'profit_per_trade': total_profit / len(optimal_trades),
            'trades_data': trades_df
        }
    
    def plot_optimal_analysis(self, df: pd.DataFrame, optimal_trades: List[OptimalTrade], 
                            performance: Dict):
        """Create comprehensive visualization of optimal trading analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'OPTIMAL Trading Analysis - {SYMBOL} (Perfect Hindsight)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Price chart with optimal trades
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['close'], label='DOGE Price', alpha=0.7, linewidth=1)
        
        for trade in optimal_trades[:100]:  # Show first 100 trades
            ax1.scatter(trade.entry_time, trade.entry_price, color='green', marker='^', s=30, alpha=0.7)
            ax1.scatter(trade.exit_time, trade.exit_price, color='red', marker='v', s=30, alpha=0.7)
        
        ax1.set_title('Optimal Entry/Exit Points')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative profit curve
        ax2 = axes[0, 1]
        if optimal_trades:
            cumulative_profits = np.cumsum([t.net_profit for t in optimal_trades])
            trade_numbers = range(1, len(optimal_trades) + 1)
            ax2.plot(trade_numbers, cumulative_profits, color='green', linewidth=2)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
        ax2.set_title('Cumulative Profit (Perfect Strategy)')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative Profit (USD)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Profit distribution
        ax3 = axes[0, 2]
        if optimal_trades:
            profit_pcts = [t.profit_pct * 100 for t in optimal_trades]
            ax3.hist(profit_pcts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=np.mean(profit_pcts), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(profit_pcts):.2f}%')
            
        ax3.set_title('Profit Distribution')
        ax3.set_xlabel('Profit per Trade (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Hold time distribution
        ax4 = axes[1, 0]
        if optimal_trades:
            hold_times = [t.hold_time_minutes for t in optimal_trades]
            ax4.hist(hold_times, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(x=np.mean(hold_times), color='red', linestyle='--',
                       label=f'Mean: {np.mean(hold_times):.1f} min')
            
        ax4.set_title('Hold Time Distribution')
        ax4.set_xlabel('Hold Time (minutes)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics
        ax5 = axes[1, 1]
        ax5.axis('off')
        if 'error' not in performance:
            metrics_text = f"""
            🎯 OPTIMAL PERFORMANCE METRICS
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Initial Balance: ${self.initial_balance:,.2f}
            Final Balance: ${performance['final_balance']:,.2f}
            Total Return: ${performance['total_profit']:,.2f} ({performance['total_return_pct']:.1%})
            Monthly Projection: {performance['monthly_return_projection']:.1%}
            
            📊 TRADE STATISTICS
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Total Trades: {performance['total_trades']}
            Avg Profit/Trade: {performance['avg_profit_pct']:.2%}
            Max Profit: {performance['max_profit_pct']:.2%}
            Avg Hold Time: {performance['avg_hold_time_minutes']:.1f} min
            Trades/Day: {performance['trades_per_day']:.1f}
            
            💸 COST ANALYSIS
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Total Fees: ${performance['total_fees_paid']:.2f}
            Fee Impact: {performance['fee_impact_pct']:.2%}
            Profit/Trade: ${performance['profit_per_trade']:.2f}
            """
            ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace')
        
        # 6. Trading frequency over time
        ax6 = axes[1, 2]
        if optimal_trades:
            # Group trades by hour of day
            trade_hours = [t.entry_time.hour for t in optimal_trades]
            ax6.hist(trade_hours, bins=24, alpha=0.7, color='purple', edgecolor='black')
            
        ax6.set_title('Trading Frequency by Hour')
        ax6.set_xlabel('Hour of Day (UTC)')
        ax6.set_ylabel('Number of Trades')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/f.kalati/Documents/crypto/optimal_trading_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_data(self, df: pd.DataFrame, optimal_trades: List[OptimalTrade]):
        """Save prepared data for AI model training."""
        print("💾 Saving training data for AI model...")
        
        # Select features for AI training
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility_5m', 'volatility_15m',
            'volume_ratio', 'price_change_1m', 'price_change_5m', 'price_change_15m',
            'trend_strength', 'volatility_percentile', 'volume_percentile'
        ]
        
        target_columns = ['is_optimal_entry', 'is_optimal_exit', 'future_profit_potential']
        
        # Prepare training dataset
        training_data = df[feature_columns + target_columns].copy()
        training_data = training_data.dropna()
        
        # Save to CSV
        training_data.to_csv('/home/f.kalati/Documents/crypto/training_data.csv')
        
        # Save optimal trades for reference
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'profit_pct': t.profit_pct,
            'net_profit': t.net_profit,
            'hold_time_minutes': t.hold_time_minutes
        } for t in optimal_trades])
        
        trades_df.to_csv('/home/f.kalati/Documents/crypto/optimal_trades.csv', index=False)
        
        print(f"   ✅ Saved {len(training_data)} training samples")
        print(f"   ✅ Saved {len(trades_df)} optimal trades")
        print(f"   📁 Files: training_data.csv, optimal_trades.csv")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """Main execution function for optimal trading analysis."""
    print("🎯 PERFECT HINDSIGHT Trading Analyzer for DOGE")
    print("=" * 70)
    print("🔮 Phase 1: COMPLETE Hindsight Analysis")
    print("📊 Finding EXACT bottoms (entry) and tops (exit) using ALL data")
    print("💰 Calculating MAXIMUM possible profit with perfect timing")
    print("🏷️ Generating optimal entry/exit labels for AI training")
    print("⚡ Using aggressive parameters for maximum trade count\n")
    
    try:
        # Initialize analyzer
        analyzer = OptimalTradingAnalyzer(INITIAL_BALANCE)
        
        # Step 1: Fetch historical data
        print("Step 1: Fetching Historical Data")
        print("-" * 40)
        df = analyzer.fetch_historical_data()
        
        if df.empty:
            print("❌ No data available for analysis")
            return
        
        print(f"📅 Data range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 Total periods: {len(df):,}")
        
        # Step 2: Calculate technical indicators
        print(f"\nStep 2: Technical Analysis")
        print("-" * 40)
        df = analyzer.calculate_technical_indicators(df)
        
        # Step 3: Find optimal trades
        print(f"\nStep 3: Optimal Trade Discovery")
        print("-" * 40)
        optimal_trades = analyzer.find_optimal_entry_exit_points(df)
        
        if not optimal_trades:
            print("❌ No profitable trades found with current parameters")
            return
        
        # Step 4: Analyze performance
        print(f"\nStep 4: Performance Analysis")
        print("-" * 40)
        performance = analyzer.analyze_optimal_performance(optimal_trades)
        
        if 'error' not in performance:
            print("\n" + "=" * 80)
            print("🎯 OPTIMAL TRADING RESULTS (Perfect Hindsight)")
            print("=" * 80)
            print(f"💰 Initial Balance: ${INITIAL_BALANCE:,.2f}")
            print(f"💰 Final Balance: ${performance['final_balance']:,.2f}")
            print(f"📈 Total Return: ${performance['total_profit']:,.2f} ({performance['total_return_pct']:.2%})")
            print(f"📅 Monthly Projection: {performance['monthly_return_projection']:.1%}")
            
            print(f"\n📊 TRADE STATISTICS")
            print("-" * 40)
            print(f"Total Optimal Trades: {performance['total_trades']}")
            print(f"Average Profit per Trade: {performance['avg_profit_pct']:.2%}")
            print(f"Maximum Single Profit: {performance['max_profit_pct']:.2%}")
            print(f"Average Hold Time: {performance['avg_hold_time_minutes']:.1f} minutes")
            print(f"Trades per Day: {performance['trades_per_day']:.1f}")
            print(f"Total Fees Paid: ${performance['total_fees_paid']:.2f}")
            
            # BENCHMARK ANALYSIS
            print(f"\n🎯 AI TRAINING BENCHMARK")
            print("=" * 60)
            if performance['monthly_return_projection'] >= 50:
                print("✅ EXCELLENT: 50%+ monthly return achieved with perfect timing!")
                print("🎯 AI Target: Try to achieve 60-80% of this performance")
            elif performance['monthly_return_projection'] >= 25:
                print("✅ GOOD: 25%+ monthly return possible with perfect timing")
                print("🎯 AI Target: Try to achieve 70-90% of this performance")
            elif performance['monthly_return_projection'] >= 10:
                print("✅ MODERATE: 10%+ monthly return possible")
                print("🎯 AI Target: Try to achieve 80%+ of this performance")
            else:
                print("⚠️ LIMITED: Low profit potential in this time period")
                print("🎯 Consider different time period or parameters")
            
            # Step 5: Create training labels
            print(f"\nStep 5: AI Training Data Preparation")
            print("-" * 40)
            df = analyzer.create_training_labels(df, optimal_trades)
            
            # Step 6: Save training data
            analyzer.save_training_data(df, optimal_trades)
            
            # Step 7: Create visualization
            print(f"\nStep 6: Visualization")
            print("-" * 40)
            analyzer.plot_optimal_analysis(df, optimal_trades, performance)
            
            print(f"\n📊 Analysis complete! Visualization saved to: optimal_trading_analysis.png")
            
            # Next steps guidance
            print(f"\n🤖 NEXT STEPS: Deep Learning Model")
            print("=" * 60)
            print("1. ✅ Perfect hindsight analysis completed")
            print("2. ✅ Training data prepared")
            print("3. 🔄 Build neural network to predict optimal points")
            print("4. 🔄 Train model on optimal entry/exit signals")
            print("5. 🔄 Test AI performance vs optimal benchmark")
            
            print(f"\n💡 Ready to build AI model that aims for:")
            print(f"   🎯 Target return: {performance['monthly_return_projection']:.1%} monthly")
            print(f"   📊 Based on {performance['total_trades']} optimal trades")
            print(f"   ⏱️ Average {performance['avg_hold_time_minutes']:.1f} minute holds")
            
        else:
            print(f"❌ Error in performance analysis: {performance['error']}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 