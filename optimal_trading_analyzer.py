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
from bisect import bisect_right
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

# Optimal trading parameters - Balanced settings for quality trades
MIN_PROFIT_PCT = 0.005  # Minimum 0.5% profit to consider a trade (quality over quantity)
MIN_PROFIT_ABSOLUTE = 5.0  # Minimum $5 profit per trade (filter out tiny gains)
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
class TradeCandidate:
    """Candidate trade for Weighted Interval Scheduling."""
    entry_idx: int
    exit_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    net_profit: float
    gross_profit: float
    fees_paid: float
    hold_minutes: float
    net_profit_pct: float
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
    
    @staticmethod
    def _compute_weight(candidate: TradeCandidate, mode: str = "rate") -> float:
        """
        محاسبه وزن برای هر کاندید معامله.
        
        Args:
            candidate: کاندید معامله
            mode: نوع وزن
                - "rate": سود بر دقیقه (بهترین کیفیت زمانی)
                - "profit": سود خالص (بیشینه سود کل)
                - "hybrid": ترکیب هر دو (70% rate + 30% profit)
        
        Returns:
            وزن محاسبه شده
        """
        if mode == "rate":
            return candidate.net_profit / max(candidate.hold_minutes, 1e-6)
        elif mode == "profit":
            return candidate.net_profit
        elif mode == "hybrid":
            rate = candidate.net_profit / max(candidate.hold_minutes, 1e-6)
            return 0.7 * rate + 0.3 * candidate.net_profit
        else:
            # Default: rate
            return candidate.net_profit / max(candidate.hold_minutes, 1e-6)
        
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
                        time.sleep(0.05)  # Binance is fast
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
        """Calculate 100+ professional technical indicators for feature engineering."""
        print("🔧 Calculating 100+ professional technical indicators...")
        
        try:
            import talib
            has_talib = True
        except ImportError:
            has_talib = False
            print("   ⚠️ TA-Lib not available, using pandas implementations")
        
        try:
            import pandas_ta as pta
            has_pandas_ta = True
        except ImportError:
            has_pandas_ta = False
            print("   ⚠️ pandas_ta not available, using manual implementations")
        
        # Store initial count
        initial_features = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        
        # Call helper methods to organize the code
        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df, has_talib)
        df = self._add_volatility_indicators(df, has_talib)
        df = self._add_volume_indicators(df)
        df = self._add_trend_strength_indicators(df, has_talib)
        df = self._add_ichimoku_indicators(df)
        df = self._add_candlestick_patterns(df)
        df = self._add_statistical_features(df)
        df = self._add_cross_features(df)
        df = self._add_time_features(df)
        
        # Clean data
        print("   🧹 Cleaning and finalizing data...")
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Check how much data we have before dropping
        print(f"   📊 Data before cleaning: {len(df)} rows")
        null_counts_before = df.isnull().sum().sum()
        print(f"   🔍 Null values before cleaning: {null_counts_before}")
        
        # Drop rows with too many NaN values (keep rows with at least 80% valid data)
        min_valid_columns = int(len(df.columns) * 0.8)
        df = df.dropna(thresh=min_valid_columns)
        
        print(f"   📊 Data after cleaning: {len(df)} rows")
        
        # Calculate final feature count
        final_features = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
        added_features = final_features - initial_features
        
        print(f"   ✅ Successfully calculated {added_features} new technical indicators!")
        print(f"   📊 Total features: {final_features} (excluding OHLCV)")
        print(f"   📈 Final dataset: {len(df):,} periods")
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend following indicators (27 features)."""
        print("   📈 Adding trend indicators...")
        
        # Simple Moving Averages (6 features)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        
        # Exponential Moving Averages (8 features)
        for period in [8, 12, 21, 26, 34, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Weighted Moving Average (3 features)
        for period in [10, 20, 50]:
            weights = np.arange(1, period + 1)
            df[f'wma_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        # Hull Moving Average (3 features)
        for period in [14, 21, 50]:
            half_length = int(period / 2)
            sqrt_length = int(np.sqrt(period))
            wma1 = df['close'].rolling(half_length).apply(
                lambda x: np.dot(x, np.arange(1, half_length + 1)) / np.sum(np.arange(1, half_length + 1)), raw=True
            )
            wma2 = df['close'].rolling(period).apply(
                lambda x: np.dot(x, np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)), raw=True
            )
            df[f'hma_{period}'] = (2 * wma1 - wma2).rolling(sqrt_length).mean()
        
        # VWAP (Volume Weighted Average Price) (1 feature)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Kaufman Adaptive Moving Average - KAMA (3 features)
        for period in [10, 20, 50]:
            change = abs(df['close'] - df['close'].shift(period))
            volatility = abs(df['close'] - df['close'].shift(1)).rolling(period).sum()
            efficiency_ratio = change / (volatility + 1e-8)
            smoothing_constant = ((efficiency_ratio * (2.0/(2+1) - 2.0/(30+1))) + 2.0/(30+1)) ** 2
            # Simple KAMA approximation using SMA when EWM fails
            df[f'kama_{period}'] = df['close'].rolling(period).mean()
        
        # Double Exponential Moving Average - DEMA (3 features)
        for period in [14, 21, 50]:
            ema1 = df['close'].ewm(span=period).mean()
            ema2 = ema1.ewm(span=period).mean()
            df[f'dema_{period}'] = 2 * ema1 - ema2
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame, has_talib: bool) -> pd.DataFrame:
        """Add momentum oscillators (18 features)."""
        print("   ⚡ Adding momentum indicators...")
        
        # RSI with multiple periods (3 features)
        for period in [6, 14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator (2 features)
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-8))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R (2 features)
        for period in [14, 21]:
            highest_high = df['high'].rolling(period).max()
            lowest_low = df['low'].rolling(period).min()
            df[f'williams_r_{period}'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low + 1e-8))
        
        # Commodity Channel Index - CCI (2 features)
        for period in [14, 20]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
        
        # Rate of Change - ROC (3 features)
        for period in [6, 12, 24]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # Money Flow Index - MFI (1 feature)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        money_ratio = positive_flow / (negative_flow + 1e-8)
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # Ultimate Oscillator (1 feature)
        if has_talib:
            try:
                import talib
                df['ultimate_osc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
            except:
                has_talib = False
        
        if not has_talib:
            bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))
            tr = np.maximum(df['high'], df['close'].shift(1)) - np.minimum(df['low'], df['close'].shift(1))
            avg7 = bp.rolling(7).sum() / (tr.rolling(7).sum() + 1e-8)
            avg14 = bp.rolling(14).sum() / (tr.rolling(14).sum() + 1e-8)
            avg28 = bp.rolling(28).sum() / (tr.rolling(28).sum() + 1e-8)
            df['ultimate_osc'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)
        
        # MACD Family (3 features)
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # TRIX - Triple Exponential Average (1 feature)
        ema1 = df['close'].ewm(span=14).mean()
        ema2 = ema1.ewm(span=14).mean()
        ema3 = ema2.ewm(span=14).mean()
        df['trix'] = ema3.pct_change() * 10000
        
        # True Strength Index - TSI (1 feature)
        momentum = df['close'].diff()
        smooth_momentum = momentum.ewm(span=25).mean().ewm(span=13).mean()
        smooth_abs_momentum = abs(momentum).ewm(span=25).mean().ewm(span=13).mean()
        df['tsi'] = 100 * smooth_momentum / (smooth_abs_momentum + 1e-8)
        
        # Awesome Oscillator (1 feature)
        median_price = (df['high'] + df['low']) / 2
        df['ao'] = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame, has_talib: bool) -> pd.DataFrame:
        """Add volatility and bands indicators (14 features)."""
        print("   📉 Adding volatility indicators...")
        
        # Average True Range - ATR (3 features)
        for period in [7, 14, 21]:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df[f'atr_{period}'] = true_range.rolling(period).mean()
        
        # Bollinger Bands (5 features)
        for period, std_dev in [(20, 2), (10, 1.5)]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (std * std_dev)
            df[f'bb_lower_{period}'] = sma - (std * std_dev)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / (sma + 1e-8)
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-8)
        
        df['bb_squeeze'] = (df['bb_width_20'] < df['bb_width_20'].rolling(20).mean() * 0.8).astype(int)
        
        # Keltner Channels (3 features)
        ema_20 = df['close'].ewm(span=20).mean()
        df['kc_upper'] = ema_20 + (2 * df['atr_14'])
        df['kc_lower'] = ema_20 - (2 * df['atr_14'])
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'] + 1e-8)
        
        # Donchian Channels (3 features)
        df['dc_upper'] = df['high'].rolling(20).max()
        df['dc_lower'] = df['low'].rolling(20).min()
        df['dc_position'] = (df['close'] - df['dc_lower']) / (df['dc_upper'] - df['dc_lower'] + 1e-8)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators (14 features)."""
        print("   📊 Adding volume indicators...")
        
        # On-Balance Volume (OBV) (1 feature)
        df['obv'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
        
        # Accumulation/Distribution Line (1 feature)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        # Chaikin Money Flow (2 features)
        for period in [10, 20]:
            mfv = clv * df['volume']
            df[f'cmf_{period}'] = mfv.rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-8)
        
        # Volume Price Trend (VPT) (1 feature)
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # Ease of Movement (2 features)
        for period in [14, 20]:
            distance_moved = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
            box_height = df['volume'] / (df['high'] - df['low'] + 1e-8)
            emv = distance_moved / (box_height + 1e-8)
            df[f'emv_{period}'] = emv.rolling(period).mean()
        
        # Volume Rate of Change (2 features)
        for period in [12, 25]:
            df[f'vroc_{period}'] = df['volume'].pct_change(period) * 100
        
        # Price Volume Trend (PVT) (1 feature)
        df['pvt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # Volume Moving Averages and Ratios (4 features)
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_sma_50'] = df['volume'].rolling(50).mean()
        df['volume_ratio_20'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
        df['volume_trend'] = (df['volume_sma_20'] > df['volume_sma_20'].shift(5)).astype(int)
        
        return df
    
    def _add_trend_strength_indicators(self, df: pd.DataFrame, has_talib: bool) -> pd.DataFrame:
        """Add trend strength indicators (12 features)."""
        print("   💪 Adding trend strength indicators...")
        
        # ADX and Directional Movement (3 features)
        high_diff = df['high'] - df['high'].shift(1)
        low_diff = df['low'].shift(1) - df['low']
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / (df['atr_14'] + 1e-8))
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / (df['atr_14'] + 1e-8))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Supertrend (2 features) - Simplified version
        for multiplier in [2, 3]:
            hl2 = (df['high'] + df['low']) / 2
            upperband = hl2 + (multiplier * df['atr_14'])
            lowerband = hl2 - (multiplier * df['atr_14'])
            
            # Simplified supertrend - use lowerband when close > upperband, else upperband
            df[f'supertrend_{multiplier}'] = np.where(df['close'] > upperband, lowerband, upperband)
        
        # Parabolic SAR (1 feature)
        if has_talib:
            try:
                import talib
                df['sar'] = talib.SAR(df['high'], df['low'])
            except:
                has_talib = False
        
        if not has_talib:
            # Simplified SAR implementation
            df['sar'] = df['low'].rolling(20).min()
        
        # Aroon Oscillator (2 features)
        for period in [14, 25]:
            aroon_up = 100 * (period - df['high'].rolling(period + 1).apply(lambda x: period - x.argmax())) / period
            aroon_down = 100 * (period - df['low'].rolling(period + 1).apply(lambda x: period - x.argmin())) / period
            df[f'aroon_{period}'] = aroon_up - aroon_down
        
        # Chande Momentum Oscillator (2 features)
        for period in [14, 20]:
            momentum = df['close'].diff()
            sum_gains = momentum.where(momentum > 0, 0).rolling(period).sum()
            sum_losses = abs(momentum.where(momentum < 0, 0)).rolling(period).sum()
            df[f'cmo_{period}'] = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses + 1e-8)
        
        # Know Sure Thing (KST) (1 feature)
        roc1 = df['close'].pct_change(10) * 100
        roc2 = df['close'].pct_change(15) * 100
        roc3 = df['close'].pct_change(20) * 100
        roc4 = df['close'].pct_change(30) * 100
        df['kst'] = (roc1.rolling(10).mean() * 1 + roc2.rolling(10).mean() * 2 + 
                    roc3.rolling(10).mean() * 3 + roc4.rolling(15).mean() * 4)
        
        # Mass Index (1 feature)
        hl_ratio = (df['high'] - df['low']) / (df['close'] + 1e-8)
        df['mass_index'] = hl_ratio.ewm(span=9).mean().ewm(span=9).mean().rolling(25).sum()
        
        return df
    
    def _add_ichimoku_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku system indicators (8 features)."""
        print("   ⛩️ Adding Ichimoku indicators...")
        
        # Tenkan-sen (Conversion Line)
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['high'].rolling(52).max()
        low_52 = df['low'].rolling(52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span) - avoid negative shift
        df['chikou_span'] = df['close'].shift(26)  # Use positive shift instead
        
        # Cloud thickness and position (3 features)
        df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])
        df['cloud_green'] = (df['senkou_span_a'] > df['senkou_span_b']).astype(int)
        df['above_cloud'] = (df['close'] > np.maximum(df['senkou_span_a'], df['senkou_span_b'])).astype(int)
        
        return df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern indicators (10 features)."""
        print("   🕯️ Adding candlestick patterns...")
        
        # Basic pattern components
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        body_ratio = body / (df['high'] - df['low'] + 1e-8)
        
        # Doji patterns (2 features)
        df['doji'] = (body < (df['high'] - df['low']) * 0.1).astype(int)
        df['dragonfly_doji'] = ((df['doji'] == 1) & (upper_shadow < body * 2) & (lower_shadow > body * 5)).astype(int)
        
        # Hammer patterns (2 features)
        df['hammer'] = ((lower_shadow > body * 2) & (upper_shadow < body) & (df['close'] > df['open'])).astype(int)
        df['hanging_man'] = ((lower_shadow > body * 2) & (upper_shadow < body) & (df['close'] < df['open'])).astype(int)
        
        # Shooting star and inverted hammer (2 features)
        df['shooting_star'] = ((upper_shadow > body * 2) & (lower_shadow < body) & (df['close'] < df['open'])).astype(int)
        df['inverted_hammer'] = ((upper_shadow > body * 2) & (lower_shadow < body) & (df['close'] > df['open'])).astype(int)
        
        # Engulfing patterns (2 features)
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                  (df['close'].shift(1) < df['open'].shift(1)) &
                                  (df['close'] > df['open'].shift(1)) & 
                                  (df['open'] < df['close'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                  (df['close'].shift(1) > df['open'].shift(1)) &
                                  (df['close'] < df['open'].shift(1)) & 
                                  (df['open'] > df['close'].shift(1))).astype(int)
        
        # Marubozu (strong candles) (2 features)
        df['white_marubozu'] = ((df['close'] > df['open']) & (body_ratio > 0.95)).astype(int)
        df['black_marubozu'] = ((df['close'] < df['open']) & (body_ratio > 0.95)).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features (8 features)."""
        print("   📊 Adding statistical features...")
        
        # Z-Scores (3 features)
        for period in [20, 50, 100]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'z_score_{period}'] = (df['close'] - mean) / (std + 1e-8)
        
        # Skewness and Kurtosis (4 features)
        for period in [20, 50]:
            df[f'skewness_{period}'] = df['close'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['close'].rolling(period).kurt()
        
        # Linear Regression Slope (1 feature)
        def linear_regression_slope(series):
            y = series.values
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
        
        df['lr_slope_20'] = df['close'].rolling(20).apply(linear_regression_slope, raw=False)
        
        return df
    
    def _add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional features (7 features)."""
        print("   🔄 Adding cross-sectional features...")
        
        # Moving Average Crosses (2 features)
        df['sma5_above_sma20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ema12_above_ema26'] = (df['ema_12'] > df['ema_26']).astype(int)
        
        # Price vs VWAP (1 feature)
        df['price_vs_vwap'] = df['close'] / (df['vwap'] + 1e-8)
        
        # ATR Ratio (1 feature)
        df['atr_ratio'] = df['atr_14'] / (df['close'] + 1e-8)
        
        # Normalized Volume (1 feature)
        volume_mean = df['volume'].rolling(50).mean()
        volume_std = df['volume'].rolling(50).std()
        df['normalized_volume'] = (df['volume'] - volume_mean) / (volume_std + 1e-8)
        
        # Momentum Divergence (1 feature)
        df['price_momentum'] = df['close'].pct_change(14)
        
        # Volatility Percentile (1 feature)
        df['volatility_percentile'] = df['atr_14'].rolling(100).rank(pct=True) * 100
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (4 features)."""
        print("   🕐 Adding time-based features...")
        
        # Hour of day (cyclical encoding) (2 features)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # Day of week (cyclical encoding) (2 features)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        return df
    
    def find_optimal_entry_exit_points(self, df: pd.DataFrame, enforce_no_overlap: bool = True, weight_mode: str = "rate") -> List[OptimalTrade]:
        """Find PERFECT entry and exit points using Weighted Interval Scheduling with DP.
        
        Algorithm:
        1. Find all local minima (valleys) for optimal entry points
        2. Find all local maxima (peaks) for optimal exit points  
        3. Build ALL candidate trades (min→max with profit > threshold)
        4. Use Dynamic Programming to select best non-overlapping subset by weight
        
        Args:
            df: DataFrame with OHLCV data
            enforce_no_overlap: If True, uses DP for optimal selection
            weight_mode: Weight metric for DP selection
                - "rate": profit per minute (best time efficiency)
                - "profit": total profit (maximize total return)
                - "hybrid": 70% rate + 30% profit
        
        Returns:
            List of optimal trades selected by DP
        """
        print(f"🎯 Finding optimal entry/exit points with DP (weight={weight_mode})...")
        print("   📊 Using complete price data to find exact tops and bottoms")
        if enforce_no_overlap:
            print(f"   🔒 Enforcing NO OVERLAP - selecting best trades by '{weight_mode}'")
        
        # Get price data
        prices = df['close'].values
        timestamps = df.index.values
        
        # ========================================================================
        # STEP 1: Find local minima (entry points) and maxima (exit points)
        # ========================================================================
        print("   🔍 Detecting local extremes...")
        
        window_size = 2  # Look 2 minutes in each direction
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
        
        # ========================================================================
        # STEP 2: Build ALL candidate trades (min → max with profit > threshold)
        # ========================================================================
        print(f"   🔄 Building all candidate trades...")
        
        candidates: List[TradeCandidate] = []
        
        for min_idx, entry_price, entry_time in local_minima:
            for max_idx, exit_price, exit_time in local_maxima:
                # Must come after entry
                if max_idx <= min_idx:
                    continue
                
                # Calculate gross profit percentage
                gross_profit_pct = (exit_price - entry_price) / entry_price
                
                # Account for fees
                net_profit_pct = gross_profit_pct - (2 * TRADING_FEE)
                
                # Check if profitable enough
                if net_profit_pct <= MIN_PROFIT_PCT:
                    continue
                
                # Make sure no other minimum between entry and exit would be better
                better_min_exists = False
                for other_min_idx, other_price, _ in local_minima:
                    if min_idx < other_min_idx < max_idx and other_price < entry_price:
                        better_min_exists = True
                        break
                
                if better_min_exists:
                    continue
                
                # Calculate trade details with fixed position size
                trade_amount = self.initial_balance * MAX_POSITION_SIZE_PCT
                quantity = trade_amount / entry_price
                
                gross_profit = quantity * (exit_price - entry_price)
                entry_fee = trade_amount * TRADING_FEE
                exit_fee = (quantity * exit_price) * TRADING_FEE
                total_fees = entry_fee + exit_fee
                net_profit = gross_profit - total_fees
                
                entry_time_dt = pd.to_datetime(entry_time)
                exit_time_dt = pd.to_datetime(exit_time)
                hold_minutes = (exit_time_dt - entry_time_dt).total_seconds() / 60.0
                
                # Check minimum hold time
                if hold_minutes < MIN_HOLD_TIME_MINUTES:
                    continue
                
                # Check minimum absolute profit (filter out tiny gains)
                if net_profit < MIN_PROFIT_ABSOLUTE:
                    continue
                
                # Create candidate
                candidates.append(TradeCandidate(
                    entry_idx=min_idx,
                    exit_idx=max_idx,
                    entry_time=entry_time_dt,
                    exit_time=exit_time_dt,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    net_profit=float(net_profit),
                    gross_profit=float(gross_profit),
                    fees_paid=float(total_fees),
                    hold_minutes=float(hold_minutes),
                    net_profit_pct=float(net_profit_pct),
                    reason=f"Candidate: {net_profit_pct:.3%}"
                ))
        
        if not candidates:
            print("   ❌ No candidate trades found")
            return []
        
        print(f"   ✅ Built {len(candidates)} candidate trades")
        
        # ========================================================================
        # STEP 3: Use Dynamic Programming to select optimal non-overlapping subset
        # ========================================================================
        if not enforce_no_overlap:
            # If overlap is allowed, just return all candidates as trades
            optimal_trades = []
            current_balance = self.initial_balance
            for c in candidates:
                trade = OptimalTrade(
                    entry_time=c.entry_time,
                    exit_time=c.exit_time,
                    entry_price=c.entry_price,
                    exit_price=c.exit_price,
                    profit_pct=c.net_profit_pct,
                    gross_profit=c.gross_profit,
                    net_profit=c.net_profit,
                    fees_paid=c.fees_paid,
                    hold_time_minutes=c.hold_minutes,
                    reason="All candidates"
                )
                optimal_trades.append(trade)
                current_balance += trade.net_profit
            
            print(f"   ✅ Returning all {len(optimal_trades)} candidates (overlap allowed)")
            return optimal_trades
        
        print(f"   🧮 Running DP to select best non-overlapping trades by '{weight_mode}'...")
        
        # Sort candidates by exit time (required for DP)
        candidates.sort(key=lambda c: c.exit_time)
        
        # Compute weights for each candidate
        weights = [self._compute_weight(c, weight_mode) for c in candidates]
        
        # Build p[i]: latest candidate that doesn't overlap with i
        # Candidate j doesn't overlap with i if: candidates[j].exit_time <= candidates[i].entry_time
        exit_times = [c.exit_time for c in candidates]
        p = []
        for i in range(len(candidates)):
            # Binary search for largest j where exit_times[j] <= entry_time[i]
            j = bisect_right(exit_times, candidates[i].entry_time) - 1
            p.append(j)
        
        # DP: dp[i] = maximum weight using candidates 0..i-1
        n = len(candidates)
        dp = [0.0] * (n + 1)
        choose = [False] * (n + 1)
        
        for i in range(1, n + 1):
            # Option 1: don't take candidate i-1
            opt1 = dp[i - 1]
            
            # Option 2: take candidate i-1
            prev_idx = p[i - 1]
            opt2 = weights[i - 1] + (dp[prev_idx + 1] if prev_idx >= 0 else 0.0)
            
            if opt2 > opt1:
                dp[i] = opt2
                choose[i] = True
            else:
                dp[i] = opt1
                choose[i] = False
        
        # Backtrack to find selected candidates
        selected: List[TradeCandidate] = []
        i = n
        while i > 0:
            if choose[i]:
                selected.append(candidates[i - 1])
                i = p[i - 1] + 1
            else:
                i -= 1
        
        selected.reverse()  # Restore chronological order
        
        print(f"   ✅ DP selected {len(selected)} optimal non-overlapping trades")
        
        # ========================================================================
        # STEP 4: Convert candidates to OptimalTrade and calculate final balance
        # ========================================================================
        optimal_trades = []
        current_balance = self.initial_balance
        
        for c in selected:
            trade = OptimalTrade(
                entry_time=c.entry_time,
                exit_time=c.exit_time,
                entry_price=c.entry_price,
                exit_price=c.exit_price,
                profit_pct=c.net_profit_pct,
                gross_profit=c.gross_profit,
                net_profit=c.net_profit,
                fees_paid=c.fees_paid,
                hold_time_minutes=c.hold_minutes,
                reason=f"DP-{weight_mode}: {c.net_profit_pct:.3%}"
            )
            optimal_trades.append(trade)
            current_balance += trade.net_profit
        
        # ========================================================================
        # STEP 5: Verify and report results
        # ========================================================================
        print(f"\n   ✅ Found {len(optimal_trades)} PERFECT trades using DP")
        print(f"   💰 Perfect trading balance: ${current_balance:,.2f}")
        print(f"   📈 Perfect return: {((current_balance - self.initial_balance) / self.initial_balance):.1%}")
        
        # Verify no overlap
        if len(optimal_trades) > 1:
            overlaps = 0
            for i in range(len(optimal_trades) - 1):
                if optimal_trades[i].exit_time > optimal_trades[i+1].entry_time:
                    overlaps += 1
            if overlaps == 0:
                print(f"   ✅ Verified: NO overlapping trades")
            else:
                print(f"   ⚠️ Warning: Found {overlaps} overlapping trades")
        
        # Calculate average metrics
        if optimal_trades:
            avg_rate = sum(t.net_profit / max(t.hold_time_minutes, 1e-6) for t in optimal_trades) / len(optimal_trades)
            avg_hold = sum(t.hold_time_minutes for t in optimal_trades) / len(optimal_trades)
            print(f"   📊 Average profit/minute: ${avg_rate:.4f}")
            print(f"   ⏱️ Average hold time: {avg_hold:.1f} minutes")
        
        return optimal_trades
    
    def _find_additional_opportunities(self, df: pd.DataFrame, main_trades: List[OptimalTrade], 
                                      existing_ranges: List[Tuple[datetime, datetime]]) -> List[OptimalTrade]:
        """Find additional profitable opportunities between main trades.
        
        Args:
            df: DataFrame with OHLCV data
            main_trades: List of main trades already found
            existing_ranges: List of (entry_time, exit_time) tuples to avoid overlap
        """
        additional_trades = []
        
        # Start with existing ranges (copy to avoid mutation)
        used_ranges = list(existing_ranges)
        
        # Add main trades ranges if not already included
        for trade in main_trades:
            if (trade.entry_time, trade.exit_time) not in used_ranges:
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
                
                # Check if the entire range [current_time, future_time] is free
                range_conflicts = False
                for start_time, end_time in used_ranges:
                    # Check for any overlap between ranges
                    if not (future_time < start_time or current_time > end_time):
                        range_conflicts = True
                        break
                
                if range_conflicts:
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
                            entry_price=float(current_price),
                            exit_price=float(future_price),
                            profit_pct=float(net_profit_pct),
                            gross_profit=float(gross_profit),
                            net_profit=float(net_profit),
                            fees_paid=float(total_fees),
                            hold_time_minutes=float(hold_time),
                            reason=f"Micro non-overlap: {net_profit_pct:.3%}"
                        )
                        
                        additional_trades.append(trade)
                        used_ranges.append((current_time, future_time))
                        break  # Found opportunity for this entry point
        
        print(f"   ✅ Found {len(additional_trades)} additional non-overlapping micro-opportunities")
        return additional_trades
    
    def create_training_labels(self, df: pd.DataFrame, optimal_trades: List[OptimalTrade]) -> pd.DataFrame:
        """Create training labels for AI model based on optimal trades.
        
        For each timestamp:
        - If it's a good entry point, mark is_optimal_entry=True
        - Store the BEST profit potential available from that point
        - If it's a good exit point, mark is_optimal_exit=True
        
        Multiple trades can start from the same point - we keep the best one.
        """
        print("🏷️ Creating training labels for AI model...")
        
        # Initialize labels
        df['is_optimal_entry'] = False
        df['is_optimal_exit'] = False
        df['future_profit_potential'] = 0.0  # Best profit potential from this point
        df['optimal_hold_time'] = 0.0  # Hold time for the best opportunity
        df['optimal_exit_price'] = 0.0  # Target exit price
        
        # Dictionary to track best profit for each entry point
        entry_profits = {}  # timestamp -> (profit_pct, hold_time, exit_price)
        exit_points = {}  # timestamp -> True
        
        # Process all trades to find best opportunities at each timestamp
        for trade in optimal_trades:
            # Ensure UTC timezone
            entry_time_utc = trade.entry_time.tz_convert('UTC') if trade.entry_time.tz is not None else trade.entry_time.tz_localize('UTC')
            exit_time_utc = trade.exit_time.tz_convert('UTC') if trade.exit_time.tz is not None else trade.exit_time.tz_localize('UTC')
            
            entry_idx = df.index.get_indexer([entry_time_utc], method='nearest')[0]
            exit_idx = df.index.get_indexer([exit_time_utc], method='nearest')[0]
            
            if entry_idx >= 0:
                entry_ts = df.index[entry_idx]
                
                # Keep the best profit for each entry point
                if entry_ts not in entry_profits or trade.profit_pct > entry_profits[entry_ts][0]:
                    entry_profits[entry_ts] = (
                        trade.profit_pct,
                        trade.hold_time_minutes,
                        trade.exit_price
                    )
            
            if exit_idx >= 0:
                exit_ts = df.index[exit_idx]
                exit_points[exit_ts] = True
        
        # Apply labels to DataFrame
        for entry_ts, (profit_pct, hold_time, exit_price) in entry_profits.items():
            try:
                idx = df.index.get_loc(entry_ts)
                df.iloc[idx, df.columns.get_loc('is_optimal_entry')] = True
                df.iloc[idx, df.columns.get_loc('future_profit_potential')] = profit_pct
                df.iloc[idx, df.columns.get_loc('optimal_hold_time')] = hold_time
                df.iloc[idx, df.columns.get_loc('optimal_exit_price')] = exit_price
            except:
                continue
        
        for exit_ts in exit_points:
            try:
                idx = df.index.get_loc(exit_ts)
                df.iloc[idx, df.columns.get_loc('is_optimal_exit')] = True
            except:
                continue
        
        # Calculate additional features for AI
        df['trend_strength'] = df['close'].rolling(10).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, raw=False
        )
        
        # Use ATR for volatility percentile (ATR is already calculated)
        if 'atr_14' in df.columns:
            df['volatility_percentile'] = df['atr_14'].rolling(100).rank(pct=True)
        else:
            # Fallback: calculate simple volatility
            df['volatility_percentile'] = df['close'].rolling(100).std().rolling(100).rank(pct=True)
        
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
        # Save in current workspace directory
        output_path = os.path.join(os.getcwd(), 'optimal_trading_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_data(self, df: pd.DataFrame, optimal_trades: List[OptimalTrade], 
                          out_path: str = "training_data.csv"):
        """Save prepared data for AI model training - SINGLE FILE with ALL features.
        
        Args:
            df: DataFrame with all features and labels
            optimal_trades: List of optimal trades (for reference only)
            out_path: Output path for training data CSV
        """
        print("💾 Saving training data for AI model...")
        
        # Make a copy to avoid modifying original
        data = df.copy()
        
        # Ensure we have a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            if "timestamp" in data.columns:
                data = data.set_index("timestamp")
            else:
                raise ValueError("Data must have a DatetimeIndex or a 'timestamp' column.")
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Ensure timezone is UTC
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        else:
            data.index = data.index.tz_convert("UTC")
        
        # Replace inf values with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Ensure label columns exist
        label_columns = {
            "is_optimal_entry": False,
            "is_optimal_exit": False,
            "future_profit_potential": 0.0,
            "optimal_hold_time": 0.0,
            "optimal_exit_price": 0.0
        }
        
        for col, default_value in label_columns.items():
            if col not in data.columns:
                data[col] = default_value
        
        # Prepare output: timestamp as first column
        output_data = data.copy()
        output_data.insert(0, "timestamp", output_data.index.astype("datetime64[ns, UTC]").astype(str))
        
        # Save to CSV (no index since timestamp is a column)
        output_data.to_csv(out_path, index=False)
        
        # Print comprehensive summary
        total_columns = len(output_data.columns)
        label_cols = ["is_optimal_entry", "is_optimal_exit", "future_profit_potential", 
                      "optimal_hold_time", "optimal_exit_price"]
        feature_columns = [col for col in output_data.columns 
                          if col not in ["timestamp"] + label_cols]
        
        print(f"   ✅ Saved {len(output_data)} training samples → {out_path}")
        print(f"   📊 Total columns: {total_columns}")
        print(f"   📊 Features: {len(feature_columns)} (includes OHLCV + all indicators)")
        print(f"   🎯 Labels: {len(label_cols)} columns for AI training:")
        print(f"      • is_optimal_entry: Entry signal (True/False)")
        print(f"      • is_optimal_exit: Exit signal (True/False)")
        print(f"      • future_profit_potential: Expected profit % from entry")
        print(f"      • optimal_hold_time: Recommended hold time (minutes)")
        print(f"      • optimal_exit_price: Target exit price")
        print(f"   📅 Timestamp: First column (ISO8601 UTC)")
        
        # Entry/exit signal counts
        entry_count = output_data["is_optimal_entry"].sum()
        exit_count = output_data["is_optimal_exit"].sum()
        avg_profit = output_data[output_data["is_optimal_entry"]]["future_profit_potential"].mean() if entry_count > 0 else 0
        avg_hold = output_data[output_data["is_optimal_entry"]]["optimal_hold_time"].mean() if entry_count > 0 else 0
        
        print(f"\n   📈 Training Data Statistics:")
        print(f"      • Entry opportunities: {entry_count} ({entry_count/len(output_data)*100:.2f}% of data)")
        print(f"      • Exit opportunities: {exit_count}")
        print(f"      • Avg profit potential: {avg_profit:.2%}")
        print(f"      • Avg hold time: {avg_hold:.1f} minutes")
        
        # Feature categories summary
        technical_columns = [col for col in feature_columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if technical_columns:
            print(f"\n   📋 Technical Indicator Categories:")
            trend_features = len([col for col in technical_columns if any(x in col for x in ['sma_', 'ema_', 'wma_', 'hma_', 'kama_', 'dema_', 'vwap'])])
            momentum_features = len([col for col in technical_columns if any(x in col for x in ['rsi_', 'stoch_', 'williams_', 'cci_', 'roc_', 'mfi', 'ultimate_', 'trix', 'tsi', 'ao', 'macd'])])
            volatility_features = len([col for col in technical_columns if any(x in col for x in ['atr_', 'bb_', 'kc_', 'dc_'])])
            volume_features = len([col for col in technical_columns if any(x in col for x in ['obv', 'ad_line', 'cmf_', 'vpt', 'emv_', 'vroc_', 'pvt', 'volume_'])])
            trend_strength_features = len([col for col in technical_columns if any(x in col for x in ['adx', 'plus_di', 'minus_di', 'supertrend_', 'sar', 'aroon_', 'cmo_', 'kst', 'mass_index'])])
            ichimoku_features = len([col for col in technical_columns if any(x in col for x in ['tenkan_', 'kijun_', 'senkou_', 'chikou_', 'cloud_', 'above_cloud'])])
            candlestick_features = len([col for col in technical_columns if any(x in col for x in ['doji', 'hammer', 'hanging', 'shooting', 'inverted', 'engulfing', 'marubozu'])])
            statistical_features = len([col for col in technical_columns if any(x in col for x in ['z_score_', 'skewness_', 'kurtosis_', 'lr_slope'])])
            cross_features = len([col for col in technical_columns if any(x in col for x in ['above', 'vs_', 'ratio', 'normalized', 'momentum', 'percentile'])])
            time_features = len([col for col in technical_columns if any(x in col for x in ['hour_', 'day_'])])
            
            print(f"   ├── Trend Following: {trend_features} features")
            print(f"   ├── Momentum: {momentum_features} features")
            print(f"   ├── Volatility: {volatility_features} features")
            print(f"   ├── Volume: {volume_features} features")
            print(f"   ├── Trend Strength: {trend_strength_features} features")
            print(f"   ├── Ichimoku: {ichimoku_features} features")
            print(f"   ├── Candlestick: {candlestick_features} features")
            print(f"   ├── Statistical: {statistical_features} features")
            print(f"   ├── Cross-sectional: {cross_features} features")
            print(f"   └── Time-based: {time_features} features")
        
        print(f"\n   🚀 Ready for AI model training with {len(technical_columns)} professional indicators!")

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
            
            print(f"\n📊 Analysis complete! Visualization saved to: {os.path.join(os.getcwd(), 'optimal_trading_analysis.png')}")
            
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