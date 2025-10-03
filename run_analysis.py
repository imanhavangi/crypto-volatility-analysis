#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Analysis Runner (No Network Calls)
==========================================
Runs analysis on cached data without any network calls.

Usage:
    python run_analysis.py
"""

import os
import sys
import pandas as pd
from optimal_trading_analyzer import OptimalTradingAnalyzer

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Configuration
DATA_PATH = "data/ohlcv_DOGEUSDT_1m.csv"
OUT_TRAIN = "training_data.csv"
OUT_TRADES = "optimal_trades.csv"
OUT_PNG = "optimal_trading_analysis.png"
INITIAL_BALANCE = 1000.0


def ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has proper DatetimeIndex."""
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def save_trades_csv(trades, path=OUT_TRADES):
    """Save optimal trades to CSV."""
    if not trades:
        print(f"⚠️ No trades to save")
        return
    
    rows = [{
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "profit_pct": t.profit_pct,
        "gross_profit": t.gross_profit,
        "net_profit": t.net_profit,
        "fees_paid": t.fees_paid,
        "hold_time_minutes": t.hold_time_minutes,
        "reason": t.reason,
    } for t in trades]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"💾 Saved {len(trades)} trades → {path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("🎯 OPTIMAL Trading Analysis (Offline Mode)")
    print("=" * 70)
    print("📂 Reading cached data (no network calls)...\n")
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found: {DATA_PATH}")
        print(f"   Please run 'python fetch_and_cache_data.py' first!")
        return
    
    try:
        # Load cached data
        print(f"📂 Loading data from {DATA_PATH}...")
        raw = pd.read_csv(DATA_PATH)
        df = ensure_dt_index(raw)
        print(f"✅ Loaded {len(df):,} rows")
        print(f"   📅 Date range: {df.index[0]} to {df.index[-1]}\n")
        
        # Initialize analyzer
        analyzer = OptimalTradingAnalyzer(initial_balance=INITIAL_BALANCE)
        
        # Step 1: Calculate technical indicators
        print("Step 1: Technical Indicators")
        print("-" * 40)
        df = analyzer.calculate_technical_indicators(df)
        
        # Step 2: Find ALL optimal entry/exit opportunities (for AI training)
        print(f"\nStep 2: Finding ALL Optimal Entry/Exit Opportunities (No Overlap Limit)")
        print("-" * 40)
        print("   ℹ️ Finding all profitable opportunities for AI training data")
        print("   ℹ️ No overlap restriction - all valid entry points will be labeled")
        # For AI training: find ALL candidates without overlap restriction
        trades = analyzer.find_optimal_entry_exit_points(df, enforce_no_overlap=False, weight_mode="profit")
        
        if not trades:
            print("❌ No profitable trades found with current parameters")
            return
        
        # Step 3: Analyze performance
        print(f"\nStep 3: Performance Analysis")
        print("-" * 40)
        perf = analyzer.analyze_optimal_performance(trades)
        
        if 'error' not in perf:
            print("\n" + "=" * 80)
            print("🎯 OPTIMAL TRADING RESULTS (Perfect Hindsight)")
            print("=" * 80)
            print(f"💰 Initial Balance: ${INITIAL_BALANCE:,.2f}")
            print(f"💰 Final Balance: ${perf['final_balance']:,.2f}")
            print(f"📈 Total Return: ${perf['total_profit']:,.2f} ({perf['total_return_pct']:.2%})")
            print(f"📅 Monthly Projection: {perf['monthly_return_projection']:.1%}")
            
            print(f"\n📊 TRADE STATISTICS")
            print("-" * 40)
            print(f"Total Optimal Trades: {perf['total_trades']}")
            print(f"Average Profit per Trade: {perf['avg_profit_pct']:.2%}")
            print(f"Maximum Single Profit: {perf['max_profit_pct']:.2%}")
            print(f"Average Hold Time: {perf['avg_hold_time_minutes']:.1f} minutes")
            print(f"Trades per Day: {perf['trades_per_day']:.1f}")
            print(f"Total Fees Paid: ${perf['total_fees_paid']:.2f}")
        
        # Step 4: Create training labels
        print(f"\nStep 4: AI Training Data Preparation")
        print("-" * 40)
        df = analyzer.create_training_labels(df, trades)
        
        # Step 5: Save training data (single file with all features)
        print(f"\nStep 5: Save Training Data")
        print("-" * 40)
        analyzer.save_training_data(df, trades, out_path=OUT_TRAIN)
        
        # Step 6: Save optimal trades CSV (optional)
        print(f"\nStep 6: Save Optimal Trades")
        print("-" * 40)
        save_trades_csv(trades, OUT_TRADES)
        
        # Step 7: Create visualization
        print(f"\nStep 7: Visualization")
        print("-" * 40)
        analyzer.plot_optimal_analysis(df, trades, perf)
        print(f"📊 Visualization saved → {OUT_PNG}")
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"📁 Output files:")
        print(f"   • {OUT_TRAIN} - Full training data with all features")
        print(f"   • {OUT_TRADES} - Optimal trades list")
        print(f"   • {OUT_PNG} - Analysis visualization")
        print(f"\n🚀 Ready for AI model training!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()

