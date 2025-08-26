# 📈 Enhanced Crypto Volatility Analysis

> A comprehensive analysis tool to identify the best cryptocurrency for scalping/day trading based on multiple factors including volatility, liquidity, spread impact, and stability metrics.

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Features

- **Multi-Factor Analysis**: Considers volatility, liquidity, spread impact, and stability
- **Real-Time Data**: Fetches live market data via CCXT library
- **Multi-Timeframe**: Analyzes 5min, 10min, 15min, and 30min intervals
- **Comprehensive Scoring**: Weighted composite score for optimal trading pair selection
- **Spread-Adjusted Metrics**: Calculates net volatility after bid-ask spread
- **Volume Analysis**: Incorporates trading volume and liquidity metrics
- **Stability Assessment**: Evaluates volatility consistency over time

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Internet connection for fetching market data

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/imanhavangi/crypto-volatility-analysis.git
   cd crypto-volatility-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

**Default Analysis (Kraken Exchange)**
```bash
python main.py
```

**Custom Exchange**
```bash
export EXCHANGE_ID=binance  # or coinbase, bitfinex, bybit, okx
python main.py
```

## 📊 Sample Output

```
🏆 RANKING SUMMARY (Exchange: KRAKEN)
------------------------------------------------------------
 rank   symbol composite_score raw_volatility net_volatility spread_pct volume_percentile
    1 DOGE/USD           0.744         0.503%         0.493%     0.010%      90%
    2  ADA/USD           0.708         0.529%         0.511%     0.017%      70%
    3  XRP/USD           0.697         0.410%         0.398%     0.012%      80%

🔍 TOP 3 DETAILED ANALYSIS
------------------------------------------------------------
#1 DOGE/USD
   Composite Score: 0.744
   Raw Volatility: 0.503%
   Net Volatility (after spread): 0.493%
   Efficiency Ratio: 0.98
   Volume Percentile: 90%
   Volatility Stability: 0.594

💡 RECOMMENDATION
------------------------------------------------------------
🚀 DOGE/USD appears to be the best choice for scalping on KRAKEN
```

## 🔧 How It Works

### Analysis Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Raw Volatility** | 25% | Price range volatility across timeframes |
| **Efficiency** | 25% | Net volatility after spread deduction |
| **Stability** | 25% | Consistency of volatility patterns |
| **Liquidity** | 25% | Trading volume and market depth |

### Methodology

1. **Data Collection**: Fetches 30 days of 1-minute OHLCV data
2. **Spread Analysis**: Estimates bid-ask spread from current market data
3. **Multi-Timeframe Resampling**: Aggregates data into 5, 10, 15, 30-minute intervals
4. **Metric Calculation**:
   - Raw volatility: `(High - Low) / Mid Price`
   - Net volatility: `Raw Volatility - Spread`
   - Stability: `1 / (1 + Coefficient of Variation)`
   - Volume percentile ranking across all symbols
5. **Composite Scoring**: Weighted combination of all metrics

## 📋 Supported Exchanges

The tool supports all exchanges available in the CCXT library:

- **Major Exchanges**: Binance, Coinbase, Kraken, Bitfinex, Bybit, OKX, Huobi
- **Regional Exchanges**: Bitstamp, Gemini, KuCoin, Gate.io
- **Full List**: [CCXT Supported Exchanges](https://github.com/ccxt/ccxt/wiki/Exchange-Markets)

## ⚙️ Configuration

### Custom Symbol Lists

Edit the `SYMBOLS` list in `main.py`:

```python
SYMBOLS = [
    "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "XRP/USD",
    "ADA/USD", "DOGE/USD", "SHIB/USD", "DOT/USD", "AVAX/USD",
    # Add your preferred trading pairs
]
```

### Adjust Scoring Weights

Modify the `WEIGHTS` dictionary to prioritize different factors:

```python
WEIGHTS = {
    'volatility': 0.25,   # Raw price volatility
    'liquidity': 0.25,    # Trading volume impact  
    'stability': 0.25,    # Volatility consistency
    'efficiency': 0.25,   # Net volatility after spread
}
```

### Custom Timeframes

Update the `TIMEFRAMES` list for different analysis intervals:

```python
TIMEFRAMES = [5, 10, 15, 30]  # Minutes
```

## 📈 Understanding the Metrics

### Composite Score (0.0 - 1.0)
Higher scores indicate better suitability for scalping/day trading.

### Raw Volatility
Average percentage price range per timeframe window.

### Net Volatility  
Volatility remaining after accounting for bid-ask spread costs.

### Efficiency Ratio
`Net Volatility / Raw Volatility` - measures how much volatility is "usable" after spread.

### Volume Percentile
Ranking of trading volume compared to other analyzed symbols (0-100%).

### Stability Score
Measures consistency of volatility patterns. Higher = more predictable.

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black main.py
```

### Type Checking
```bash
mypy main.py
```

## 📚 Dependencies

- **ccxt**: Cryptocurrency exchange trading library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **typing**: Type hints (Python 3.8+)

## ⚠️ Disclaimers

- **Not Financial Advice**: This tool is for educational and research purposes only
- **Market Risk**: Cryptocurrency trading involves substantial risk of loss
- **Data Accuracy**: Market data accuracy depends on exchange API reliability
- **Past Performance**: Historical volatility doesn't guarantee future results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) for the excellent exchange integration library
- [Pandas](https://pandas.pydata.org/) for powerful data analysis capabilities
- Cryptocurrency exchanges for providing market data APIs

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/imanhavangi/crypto-volatility-analysis/issues) section
2. Create a new issue with detailed description
3. Include error messages and system information

---

**⭐ Star this repository if you find it useful!** 