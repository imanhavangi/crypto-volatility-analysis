#!/usr/bin/env python3
"""
Example usage of Enhanced Crypto Volatility Analysis
===================================================
This file demonstrates different ways to use the analysis tool
with custom configurations and settings.
"""

import os
from main import rank_symbols_comprehensive, print_comprehensive_report

def example_custom_symbols():
    """Example: Analyze custom list of symbols"""
    print("📊 Example 1: Custom Symbol Analysis")
    print("=" * 50)
    
    # Custom symbol list focusing on major cryptocurrencies
    custom_symbols = [
        "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "MATIC/USD"
    ]
    
    # You can modify the global SYMBOLS list in main.py
    # or create your own analysis function
    
    print(f"Analyzing: {', '.join(custom_symbols)}")
    print("This would analyze only the specified symbols...\n")


def example_different_exchange():
    """Example: Using different exchange"""
    print("📊 Example 2: Different Exchange Analysis")
    print("=" * 50)
    
    # Set environment variable for different exchange
    os.environ["EXCHANGE_ID"] = "binance"  # or "coinbase", "bitfinex", etc.
    
    print("Exchange set to: Binance")
    print("Run: python main.py")
    print("This will analyze the same symbols on Binance instead of Kraken\n")


def example_interpretation():
    """Example: How to interpret results"""
    print("📊 Example 3: Interpreting Results")
    print("=" * 50)
    
    print("""
When you see results like:
    
 rank   symbol composite_score raw_volatility net_volatility spread_pct volume_percentile
    1 DOGE/USD           0.744         0.503%         0.493%     0.010%      90%
    
This means:
✅ DOGE/USD has the highest overall score (0.744)
✅ Raw volatility is 0.503% per 10-minute window
✅ After spread costs, usable volatility is 0.493%
✅ Spread impact is minimal (0.010%)
✅ Volume is in the 90th percentile (very liquid)

For scalping/day trading, you want:
- High composite score (>0.6)
- Good net volatility (>0.3%)
- Low spread impact (<0.05%)
- High volume percentile (>50%)
- Reasonable stability (>0.5)
""")


def example_risk_considerations():
    """Example: Risk management considerations"""
    print("📊 Example 4: Risk Management")
    print("=" * 50)
    
    print("""
⚠️  Important Risk Considerations:

1. Volatility Stability < 0.6:
   → Use smaller position sizes
   → Implement stricter stop-losses
   → Monitor for sudden volatility spikes

2. High Spread (>0.05%):
   → Reduces profit margins
   → May require wider profit targets
   → Consider transaction costs carefully

3. Low Volume Percentile (<30%):
   → Risk of slippage on larger orders
   → May have difficulty exiting positions quickly
   → Consider reducing position size

4. Market Conditions:
   → This analysis is based on historical data
   → Current market sentiment may differ
   → Always use proper risk management
""")


if __name__ == "__main__":
    print("🚀 Enhanced Crypto Volatility Analysis - Examples")
    print("=" * 60)
    print()
    
    example_custom_symbols()
    example_different_exchange() 
    example_interpretation()
    example_risk_considerations()
    
    print("💡 To run the full analysis:")
    print("   python main.py")
    print()
    print("💡 To use a different exchange:")
    print("   export EXCHANGE_ID=binance && python main.py")
    print()
    print("📖 See README.md for more detailed documentation") 