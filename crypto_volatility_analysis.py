#!/usr/bin/env python3
"""
Enhanced Crypto Volatility Analysis v2.0
========================================
A comprehensive analysis tool to identify the best cryptocurrency for scalping
based on multiple factors including:
- Volatility relative to spread
- Liquidity metrics (volume and depth)  
- Volatility stability and distribution
- Multi-timeframe analysis
- Comprehensive scoring system

Dependencies:
    pip install ccxt pandas numpy

Usage:
    export EXCHANGE_ID=kraken          # or coinbase, bitfinex, bybit, okx …
    python main.py
"""

from __future__ import annotations
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import ccxt

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOLS = [
    "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "XRP/USD",
    "ADA/USD", "DOGE/USD", "SHIB/USD", "DOT/USD", "AVAX/USD",
]

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")
RATE_LIMIT_BUFFER = 0.1

# Analysis timeframes (in minutes)
TIMEFRAMES = [1]

# Scoring weights for different factors
WEIGHTS = {
    'volatility': 0.25,      # Raw volatility
    'liquidity': 0.25,       # Volume-adjusted volatility  
    'stability': 0.25,       # Volatility consistency
    'efficiency': 0.25,      # Net volatility after spread
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_exchange(exchange_id: str) -> ccxt.Exchange:
    """Initialize exchange with rate limiting."""
    if exchange_id not in ccxt.exchanges:
        raise ValueError(f"Unknown exchange id '{exchange_id}'.")
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def setup_exchange_with_fallback(symbol: str) -> Tuple[ccxt.Exchange, str]:
    """Try multiple exchanges to find one that works for the symbol."""
    # Exchange configurations with symbol mappings
    exchanges_to_try = [
        ('binance', {
            'enableRateLimit': True,
            'rateLimit': 1200,
            'timeout': 30000,
            'options': {'adjustForTimeDifference': True}
        }, 'USDT'),
        ('okx', {
            'enableRateLimit': True,
            'rateLimit': 2000,
            'timeout': 30000,
        }, 'USDT'),
        ('kraken', {
            'enableRateLimit': True,
            'rateLimit': 3000,
            'timeout': 60000,
        }, 'USD'),
        ('coinbase', {
            'enableRateLimit': True,
            'rateLimit': 10000,
            'timeout': 60000,
        }, 'USD'),
    ]
    
    base_symbol = symbol.split('/')[0]  # Extract base currency (e.g., BTC from BTC/USD)
    
    for exchange_name, config, quote_currency in exchanges_to_try:
        try:
            print(f"   🔗 Trying {exchange_name.upper()}...")
            
            # Create symbol for this exchange
            if exchange_name in ['binance', 'okx']:
                symbol_to_use = f"{base_symbol}/{quote_currency}"  # e.g., BTC/USDT
            else:  # kraken, coinbase
                symbol_to_use = f"{base_symbol}/USD"  # e.g., BTC/USD
            
            exchange = getattr(ccxt, exchange_name)(config)
            exchange.load_markets()
            
            # Test if symbol exists
            if symbol_to_use in exchange.markets:
                print(f"   ✅ Connected to {exchange_name.upper()} with {symbol_to_use}")
                return exchange, symbol_to_use
            else:
                print(f"   ❌ {symbol_to_use} not available on {exchange_name.upper()}")
                
        except Exception as e:
            print(f"   ❌ Failed to connect to {exchange_name.upper()}: {e}")
            continue
    
    # Fallback to original exchange if all fail
    print(f"   🔄 Falling back to original exchange: {EXCHANGE_ID}")
    return setup_exchange(EXCHANGE_ID), symbol


def fetch_ohlcv_with_volume(exchange: ccxt.Exchange, symbol: str, 
                           start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch comprehensive OHLCV data with advanced retry and deduplication logic."""
    print(f"   ↳ Fetching data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    # Calculate target candles for the period
    total_minutes = int((end - start).total_seconds() / 60)
    target_candles = total_minutes
    timeframe = "1m"
    
    # Exchange-specific optimizations
    exchange_id = exchange.id.lower()
    if exchange_id == 'binance':
        batch_size = 1000
        base_delay = 0.5
    elif exchange_id == 'okx':
        batch_size = 1000  
        base_delay = 0.8
    elif exchange_id == 'kraken':
        batch_size = 720
        base_delay = 3.0
    else:
        batch_size = 500
        base_delay = 1.5
    
    print(f"   🎯 Target: {target_candles:,} candles")
    print(f"   📦 Batch size: {batch_size} candles per request")
    
    all_data = []
    current_end_ms = int(end.timestamp() * 1000)
    batch_count = 0
    max_batches = (target_candles // batch_size) + 10
    consecutive_empty_batches = 0
    
    while len(all_data) < target_candles and batch_count < max_batches:
        batch_count += 1
        retry_count = 0
        max_retries = 3
        batch_success = False
        
        while not batch_success and retry_count < max_retries:
            try:
                # Calculate time window for this batch
                batch_start_ms = current_end_ms - (batch_size * 60 * 1000)
                
                # Ensure we don't go before start time
                if batch_start_ms < int(start.timestamp() * 1000):
                    batch_start_ms = int(start.timestamp() * 1000)
                
                ohlcv = exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe, 
                    since=batch_start_ms, 
                    limit=batch_size
                )
                
                if not ohlcv or len(ohlcv) < 10:
                    consecutive_empty_batches += 1
                    if consecutive_empty_batches >= 3:
                        print(f"\n   🛑 Stopping after {consecutive_empty_batches} consecutive empty batches")
                        batch_success = True
                        break
                    # Go back further if no data
                    current_end_ms = batch_start_ms - (batch_size * 60 * 1000)
                    batch_success = True
                    continue
                
                # Remove duplicates based on timestamp
                unique_ohlcv = []
                seen_timestamps = set()
                for candle in ohlcv:
                    if candle[0] not in seen_timestamps:
                        unique_ohlcv.append(candle)
                        seen_timestamps.add(candle[0])
                
                # Add only new candles we don't already have
                new_candles = []
                existing_timestamps = {candle[0] for candle in all_data} if all_data else set()
                
                for candle in unique_ohlcv:
                    if candle[0] not in existing_timestamps:
                        new_candles.append(candle)
                
                # Add to beginning of list (going backwards in time)
                all_data = new_candles + all_data
                
                if ohlcv:
                    first_candle_time = datetime.fromtimestamp(ohlcv[0][0]/1000, timezone.utc)
                    last_candle_time = datetime.fromtimestamp(ohlcv[-1][0]/1000, timezone.utc)
                    
                    print(f"\n   ✅ Batch {batch_count}: {len(ohlcv)} candles ({len(new_candles)} new)")
                    print(f"      📅 Time range: {first_candle_time.strftime('%H:%M')} to {last_candle_time.strftime('%H:%M')}")
                    print(f"      📈 Total collected: {len(all_data):,} candles")
                    
                    progress = min(len(all_data) / target_candles, 1.0)
                    print(f"      🎯 Progress: {progress:.1%}")
                
                # Reset consecutive empty counter on success
                if len(new_candles) > 0:
                    consecutive_empty_batches = 0
                    current_end_ms = ohlcv[0][0] - 60000  # Start 1 minute before first candle
                else:
                    consecutive_empty_batches += 1
                    current_end_ms = batch_start_ms - (batch_size * 60 * 1000)
                
                batch_success = True
                
                # Exchange-specific delays
                time.sleep(base_delay)
                
                # Stop if we've collected enough data
                if len(all_data) >= target_candles:
                    print(f"\n   ✅ Target reached! Collected {len(all_data):,} candles")
                    break
                    
            except Exception as e:
                retry_count += 1
                print(f"\n   ⚠️ Error in batch {batch_count}, attempt {retry_count}/{max_retries}: {e}")
                
                if retry_count < max_retries:
                    wait_time = base_delay * retry_count * 2  # Progressive backoff
                    print(f"   ⏳ Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Failed to fetch batch {batch_count} after {max_retries} attempts, skipping...")
                    batch_success = True
        
        # Check if we should break out of main loop
        if consecutive_empty_batches >= 3:
            break
        
        # Safety check - don't go before start time
        if current_end_ms <= int(start.timestamp() * 1000):
            print(f"   🏁 Reached start time boundary")
            break

    print(f"\n   ✅ Fetched {len(all_data):,} candles total")

    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    
    # Remove duplicate timestamps
    print(f"   🔍 Checking for duplicate timestamps...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['ts'], keep='first')
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"   🧹 Removed {initial_count - final_count} duplicate timestamps")
    
    df.set_index("ts", inplace=True)
    
    # Convert to float and remove any invalid data
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Sort by timestamp to ensure proper order
    df = df.sort_index()
    
    print(f"   ✅ Final dataset: {len(df):,} unique candles")
    print(f"   📅 Date range: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    
    return df


def get_spread_estimate(exchange: ccxt.Exchange, symbol: str) -> float:
    """Estimate bid-ask spread as percentage of mid price."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        if ticker['bid'] and ticker['ask']:
            spread = ticker['ask'] - ticker['bid'] 
            mid = (ticker['bid'] + ticker['ask']) / 2
            return spread / mid if mid > 0 else 0.001  # Default 0.1% if no data
        return 0.001  # Default spread
    except Exception:
        return 0.001  # Default spread


def compute_comprehensive_metrics(df: pd.DataFrame, spread_pct: float, 
                                timeframe_min: int) -> Dict[str, float]:
    """Compute comprehensive volatility and liquidity metrics."""
    if df.empty:
        return {}
    
    timeframe_str = f"{timeframe_min}min"
    
    # Resample data
    resampled = df.resample(timeframe_str).agg({
        'high': 'max',
        'low': 'min', 
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if len(resampled) < 2:
        return {}
    
    # Calculate mid price and raw volatility
    mid = (resampled['high'] + resampled['low']) / 2
    raw_volatility = ((resampled['high'] - resampled['low']) / mid).dropna()
    
    # Net volatility after spread
    net_volatility = (raw_volatility - spread_pct).clip(lower=0)
    
    # Volume-weighted volatility  
    volume_weights = resampled['volume'] / resampled['volume'].sum()
    volume_weighted_vol = (raw_volatility * volume_weights).sum()
    
    # Volatility stability metrics
    vol_std = raw_volatility.std()
    vol_cv = vol_std / raw_volatility.mean() if raw_volatility.mean() > 0 else float('inf')
    
    # Liquidity score (normalized volume)
    avg_volume = resampled['volume'].mean()
    
    # Returns-based volatility for comparison
    returns = resampled['close'].pct_change().dropna()
    returns_vol = returns.std() if len(returns) > 0 else 0
    
    return {
        'raw_volatility': raw_volatility.mean(),
        'net_volatility': net_volatility.mean(), 
        'volume_weighted_volatility': volume_weighted_vol,
        'volatility_stability': 1 / (1 + vol_cv),  # Higher is more stable
        'avg_volume': avg_volume,
        'returns_volatility': returns_vol,
        'spread_pct': spread_pct,
        'data_points': len(resampled)
    }


def calculate_composite_score(metrics: Dict[str, float], 
                            volume_percentile: float) -> float:
    """Calculate a composite score for scalping suitability."""
    if not metrics:
        return 0.0
    
    # Normalize metrics (0-1 scale)
    volatility_score = min(metrics['raw_volatility'] * 100, 1.0)  # Cap at 1% = score 1
    efficiency_score = metrics['net_volatility'] / (metrics['raw_volatility'] + 1e-8)
    stability_score = metrics['volatility_stability']
    liquidity_score = volume_percentile / 100.0  # Convert percentile to 0-1
    
    # Weighted composite score
    composite = (
        WEIGHTS['volatility'] * volatility_score +
        WEIGHTS['efficiency'] * efficiency_score + 
        WEIGHTS['stability'] * stability_score +
        WEIGHTS['liquidity'] * liquidity_score
    )
    
    return composite


def analyze_symbol_with_fallback(symbol: str, 
                               start: datetime, end: datetime) -> Dict[str, any]:
    """Comprehensive analysis of a single symbol with exchange fallback."""
    print(f"📊 Analyzing {symbol}...")
    
    # Try to get the best exchange for this symbol
    try:
        exchange, actual_symbol = setup_exchange_with_fallback(symbol)
    except Exception as e:
        print(f"   ❌ Could not setup any exchange for {symbol}: {e}")
        return {}
    
    # Fetch data
    df = fetch_ohlcv_with_volume(exchange, actual_symbol, start, end)
    if df.empty:
        print(f"   ❌ No data available for {actual_symbol} on {exchange.id.upper()}")
        return {}
    
    # Get spread estimate
    spread_pct = get_spread_estimate(exchange, actual_symbol)
    print(f"   📏 Estimated spread: {spread_pct:.4%}")
    
    # Multi-timeframe analysis
    timeframe_results = {}
    for tf in TIMEFRAMES:
        metrics = compute_comprehensive_metrics(df, spread_pct, tf)
        if metrics:
            timeframe_results[f"{tf}min"] = metrics
    
    if not timeframe_results:
        return {}
    
    # Calculate average volume for percentile ranking
    avg_volume = np.mean([m['avg_volume'] for m in timeframe_results.values()])
    
    return {
        'symbol': symbol,  # Keep original symbol for consistency
        'actual_symbol': actual_symbol,  # Store the actual symbol used
        'exchange_used': exchange.id,
        'spread_pct': spread_pct,
        'avg_volume': avg_volume,
        'timeframe_metrics': timeframe_results,
        'data_quality': len(df)
    }


def analyze_symbol(exchange: ccxt.Exchange, symbol: str, 
                  start: datetime, end: datetime) -> Dict[str, any]:
    """Comprehensive analysis of a single symbol (legacy version)."""
    print(f"📊 Analyzing {symbol}...")
    
    # Fetch data
    df = fetch_ohlcv_with_volume(exchange, symbol, start, end)
    if df.empty:
        print(f"   ❌ No data available for {symbol}")
        return {}
    
    # Get spread estimate
    spread_pct = get_spread_estimate(exchange, symbol)
    print(f"   📏 Estimated spread: {spread_pct:.4%}")
    
    # Multi-timeframe analysis
    timeframe_results = {}
    for tf in TIMEFRAMES:
        metrics = compute_comprehensive_metrics(df, spread_pct, tf)
        if metrics:
            timeframe_results[f"{tf}min"] = metrics
    
    if not timeframe_results:
        return {}
    
    # Calculate average volume for percentile ranking
    avg_volume = np.mean([m['avg_volume'] for m in timeframe_results.values()])
    
    return {
        'symbol': symbol,
        'spread_pct': spread_pct,
        'avg_volume': avg_volume,
        'timeframe_metrics': timeframe_results,
        'data_quality': len(df)
    }


def rank_symbols_comprehensive(symbols: List[str]) -> pd.DataFrame:
    """Comprehensive ranking of symbols for scalping suitability with exchange fallback."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    
    print(f"🔍 Auto-selecting best exchanges for each symbol")
    print(f"📅 Analysis period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"⏱️  Timeframes: {', '.join([f'{tf}min' for tf in TIMEFRAMES])}")
    print("=" * 60)
    
    # Analyze all symbols with exchange fallback
    results = []
    for sym in symbols:
        analysis = analyze_symbol_with_fallback(sym, start, end)
        if analysis:
            results.append(analysis)
        print()
    
    if not results:
        print("❌ No data available for any symbols")
        return pd.DataFrame()
    
    # Show exchange summary
    exchange_usage = {}
    for result in results:
        exchange_used = result.get('exchange_used', 'unknown')
        if exchange_used not in exchange_usage:
            exchange_usage[exchange_used] = []
        exchange_usage[exchange_used].append(result['symbol'])
    
    print("📊 Exchange Usage Summary:")
    print("-" * 40)
    for exchange, symbols_list in exchange_usage.items():
        print(f"   {exchange.upper()}: {', '.join(symbols_list)}")
    print()
    
    # Calculate volume percentiles for liquidity scoring
    volumes = [r['avg_volume'] for r in results]
    volume_percentiles = {}
    for i, result in enumerate(results):
        percentile = (np.sum(np.array(volumes) <= volumes[i]) / len(volumes)) * 100
        volume_percentiles[result['symbol']] = percentile
    
    # Create comprehensive ranking
    ranking_data = []
    for result in results:
        symbol = result['symbol']
        
        # Use 1-minute timeframe as primary (updated from original code)
        primary_tf = "1min"
        if primary_tf not in result['timeframe_metrics']:
            primary_tf = list(result['timeframe_metrics'].keys())[0]
        
        metrics = result['timeframe_metrics'][primary_tf]
        vol_percentile = volume_percentiles[symbol]
        
        composite_score = calculate_composite_score(metrics, vol_percentile)
        
        ranking_data.append({
            'symbol': symbol,
            'exchange': result.get('exchange_used', 'unknown'),
            'composite_score': composite_score,
            'raw_volatility': metrics['raw_volatility'],
            'net_volatility': metrics['net_volatility'], 
            'spread_pct': result['spread_pct'],
            'volatility_stability': metrics['volatility_stability'],
            'volume_percentile': vol_percentile,
            'avg_volume': result['avg_volume'],
            'efficiency_ratio': metrics['net_volatility'] / (metrics['raw_volatility'] + 1e-8)
        })
    
    # Create and sort DataFrame
    df = pd.DataFrame(ranking_data)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def print_comprehensive_report(ranked_df: pd.DataFrame):
    """Print detailed analysis report."""
    if ranked_df.empty:
        print("❌ No data available for analysis.")
        return
    
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE CRYPTO SCALPING ANALYSIS REPORT")
    print("=" * 80)
    
    # Summary table
    print(f"\n🏆 RANKING SUMMARY (Multi-Exchange Analysis)")
    print("-" * 60)
    
    display_df = ranked_df[['rank', 'symbol', 'exchange', 'composite_score', 'raw_volatility', 
                           'net_volatility', 'spread_pct', 'volume_percentile']].copy()
    
    # Format for display
    display_df['composite_score'] = display_df['composite_score'].apply(lambda x: f"{x:.3f}")
    display_df['raw_volatility'] = display_df['raw_volatility'].apply(lambda x: f"{x:.3%}")
    display_df['net_volatility'] = display_df['net_volatility'].apply(lambda x: f"{x:.3%}")
    display_df['spread_pct'] = display_df['spread_pct'].apply(lambda x: f"{x:.3%}")
    display_df['volume_percentile'] = display_df['volume_percentile'].apply(lambda x: f"{x:.0f}%")
    
    print(display_df.to_string(index=False))
    
    # Top 3 detailed analysis
    print(f"\n🔍 TOP 3 DETAILED ANALYSIS")
    print("-" * 60)
    
    for i in range(min(3, len(ranked_df))):
        row = ranked_df.iloc[i]
        print(f"\n#{i+1} {row['symbol']} (via {row['exchange'].upper()})")
        print(f"   Composite Score: {row['composite_score']:.3f}")
        print(f"   Raw Volatility: {row['raw_volatility']:.3%}")
        print(f"   Net Volatility (after spread): {row['net_volatility']:.3%}")
        print(f"   Spread Impact: {row['spread_pct']:.3%}")
        print(f"   Efficiency Ratio: {row['efficiency_ratio']:.2f}")
        print(f"   Volume Percentile: {row['volume_percentile']:.0f}%")
        print(f"   Volatility Stability: {row['volatility_stability']:.3f}")
    
    # Recommendations
    best = ranked_df.iloc[0]
    print(f"\n💡 RECOMMENDATION")
    print("-" * 60)
    print(f"🚀 {best['symbol']} appears to be the best choice for scalping via {best['exchange'].upper()}:")
    print(f"   • Highest composite score ({best['composite_score']:.3f})")
    print(f"   • Effective volatility after spread: {best['net_volatility']:.3%}")
    print(f"   • Volume ranking: {best['volume_percentile']:.0f}th percentile")
    
    if best['spread_pct'] > 0.002:  # 0.2%
        print(f"   ⚠️  Note: Spread is relatively high ({best['spread_pct']:.3%})")
    
    if best['volatility_stability'] < 0.7:
        print(f"   ⚠️  Note: Volatility may be inconsistent (stability: {best['volatility_stability']:.3f})")
    
    print(f"\n📊 Analysis completed using {len(ranked_df)} symbols over 30 days")
    print("=" * 80)


def main() -> None:
    """Main execution function."""
    try:
        print("🔄 Starting Enhanced Crypto Volatility Analysis...")
        
        ranked = rank_symbols_comprehensive(SYMBOLS)
        print_comprehensive_report(ranked)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
