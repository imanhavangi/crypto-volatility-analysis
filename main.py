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
TIMEFRAMES = [5, 10, 15, 30]

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


def fetch_ohlcv_with_volume(exchange: ccxt.Exchange, symbol: str, 
                           start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV data including volume for comprehensive analysis."""
    all_rows: list[list] = []
    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    timeframe = "1m"
    limit = 1000

    print(f"   ↳ Fetching data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    while since_ms < end_ms:
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not chunk:
                break
            all_rows.extend(chunk)
            since_ms = chunk[-1][0] + 60_000  # advance by one minute
            time.sleep((exchange.rateLimit / 1000) + RATE_LIMIT_BUFFER)
        except Exception as e:
            print(f"   ⚠️  Error fetching data: {e}")
            break

    if not all_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    
    # Convert to float and remove any invalid data
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
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


def analyze_symbol(exchange: ccxt.Exchange, symbol: str, 
                  start: datetime, end: datetime) -> Dict[str, any]:
    """Comprehensive analysis of a single symbol."""
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
    """Comprehensive ranking of symbols for scalping suitability."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    exchange = setup_exchange(EXCHANGE_ID)
    
    print(f"🏪 Using exchange: {exchange.id}")
    print(f"📅 Analysis period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"⏱️  Timeframes: {', '.join([f'{tf}min' for tf in TIMEFRAMES])}")
    print("=" * 60)
    
    exchange.load_markets()
    
    if not exchange.has.get("fetchOHLCV", False):
        raise RuntimeError(f"Exchange '{exchange.id}' does not support OHLCV.")
    
    # Analyze all symbols
    results = []
    for sym in symbols:
        if sym not in exchange.symbols:
            print(f"⚠️  {sym} not available on {exchange.id}; skipping.")
            continue
            
        analysis = analyze_symbol(exchange, sym, start, end)
        if analysis:
            results.append(analysis)
        print()
    
    if not results:
        return pd.DataFrame()
    
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
        
        # Use 10-minute timeframe as primary (most common for scalping)
        primary_tf = "10min"
        if primary_tf not in result['timeframe_metrics']:
            primary_tf = list(result['timeframe_metrics'].keys())[0]
        
        metrics = result['timeframe_metrics'][primary_tf]
        vol_percentile = volume_percentiles[symbol]
        
        composite_score = calculate_composite_score(metrics, vol_percentile)
        
        ranking_data.append({
            'symbol': symbol,
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
    print(f"\n🏆 RANKING SUMMARY (Exchange: {EXCHANGE_ID.upper()})")
    print("-" * 60)
    
    display_df = ranked_df[['rank', 'symbol', 'composite_score', 'raw_volatility', 
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
        print(f"\n#{i+1} {row['symbol']}")
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
    print(f"🚀 {best['symbol']} appears to be the best choice for scalping on {EXCHANGE_ID.upper()}:")
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
