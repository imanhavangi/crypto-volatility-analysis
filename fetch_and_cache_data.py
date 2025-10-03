#!/usr/bin/env python3
"""
Data Fetcher and Cache Manager for DOGE/USDT
=============================================
Fetches OHLCV data from exchange and caches it locally with incremental updates.

Usage:
    python fetch_and_cache_data.py
"""

import os
from datetime import datetime, timedelta, timezone
import ccxt
import pandas as pd
import time

# Configuration
EXCHANGE_NAME = "binance"
SYMBOLS = {
    "binance": "DOGE/USDT",
    "okx": "DOGE/USDT",
    "kraken": "DOGE/USD"
}
TIMEFRAME = "1m"
DAYS_BACK = 30  # For initial fetch
OUT_PATH = "data/ohlcv_DOGEUSDT_1m.csv"
LIMIT = 1000  # Max candles per request

# Ensure data directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def load_exchange(name: str):
    """Initialize and load exchange."""
    print(f"📡 Connecting to {name.upper()}...")
    ex = getattr(ccxt, name)({
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {"adjustForTimeDifference": True} if name == "binance" else {}
    })
    ex.load_markets()
    print(f"✅ Connected to {name.upper()}")
    return ex


def pick_symbol(ex, exchange_name: str) -> str:
    """Select appropriate symbol for exchange."""
    sym = SYMBOLS.get(exchange_name, "DOGE/USDT")
    if sym not in ex.markets:
        raise ValueError(f"❌ Symbol {sym} not found on {exchange_name}")
    print(f"🎯 Using symbol: {sym}")
    return sym


def timeframe_ms(tf: str) -> int:
    """Convert timeframe string to milliseconds."""
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000
    }
    return mapping.get(tf, 60_000)


def read_existing():
    """Read existing cached data and return last timestamp."""
    if not os.path.exists(OUT_PATH):
        print("📂 No existing cache found, will fetch fresh data")
        return None, None
    
    print("📂 Reading existing cache...")
    df = pd.read_csv(OUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    
    last_ts_ms = int(df["timestamp"].iloc[-1].timestamp() * 1000)
    print(f"✅ Existing cache: {len(df):,} rows")
    print(f"   Last timestamp: {df['timestamp'].iloc[-1]}")
    
    return df, last_ts_ms


def save_ohlcv(df: pd.DataFrame):
    """Save OHLCV data to CSV."""
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df.to_csv(OUT_PATH, index=False)
    print(f"💾 Saved {len(df):,} rows → {OUT_PATH}")


def fetch_ohlcv_incremental(ex, symbol: str, since_ms: int, until_ms: int):
    """Fetch OHLCV data incrementally from since_ms to until_ms."""
    all_rows = []
    tf_ms = timeframe_ms(TIMEFRAME)
    cursor = since_ms
    batch_count = 0
    
    print(f"📊 Fetching data from {datetime.utcfromtimestamp(since_ms/1000)} to {datetime.utcfromtimestamp(until_ms/1000)}")
    
    while cursor < until_ms:
        batch_count += 1
        try:
            batch = ex.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=cursor,
                limit=LIMIT
            )
            
            if not batch:
                print(f"   ⚠️ No more data available")
                break
            
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            
            # Progress update
            progress = len(all_rows)
            print(f"\r   📦 Batch {batch_count}: {len(batch)} candles | Total: {progress:,}", end="", flush=True)
            
            # Move cursor forward
            cursor = last_ts + tf_ms
            
            # Stop if we got less than limit or reached end
            if len(batch) < LIMIT and (last_ts + tf_ms) >= until_ms:
                break
            
            # Small delay to respect rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\n   ❌ Error fetching batch {batch_count}: {e}")
            print(f"   ⏳ Waiting 3 seconds before retry...")
            time.sleep(3)
            continue
    
    print()  # New line after progress
    
    if not all_rows:
        print("⚠️ No new data fetched")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # Convert to DataFrame
    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=["timestamp"])
    if len(df) < initial_count:
        print(f"   🧹 Removed {initial_count - len(df)} duplicate timestamps")
    
    return df


def main():
    """Main execution function."""
    print("=" * 70)
    print("🚀 Data Fetcher and Cache Manager")
    print("=" * 70)
    
    try:
        # Load exchange
        ex = load_exchange(EXCHANGE_NAME)
        symbol = pick_symbol(ex, EXCHANGE_NAME)
        
        # Check existing data
        existing, last_ms = read_existing()
        
        # Determine fetch range
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        
        if last_ms is not None:
            # Incremental update
            start_ms = last_ms + 1
            print(f"\n🔄 Incremental update mode")
        else:
            # Fresh fetch
            start_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=DAYS_BACK)).timestamp() * 1000)
            print(f"\n🆕 Fresh fetch mode ({DAYS_BACK} days)")
        
        # Fetch new data
        new_df = fetch_ohlcv_incremental(ex, symbol, start_ms, now_ms)
        
        # Merge and save
        if existing is None:
            final_df = new_df
        else:
            final_df = pd.concat([existing, new_df], ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        
        if len(final_df) > 0:
            save_ohlcv(final_df)
            print(f"\n✅ Cache updated successfully!")
            print(f"   📅 Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
            print(f"   📊 Total candles: {len(final_df):,}")
        else:
            print("\n⚠️ No data to save")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

