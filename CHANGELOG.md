# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-08-26

### 🚀 Major Release - Complete Rewrite

This is a complete rewrite of the original volatility analysis tool with comprehensive improvements.

### ✨ Added
- **Multi-Factor Analysis**: Now considers 4 key factors with weighted scoring
  - Raw volatility (25%)
  - Liquidity/Volume (25%)
  - Volatility stability (25%)
  - Efficiency after spread (25%)
- **Spread Analysis**: Real-time bid-ask spread estimation and impact calculation
- **Multi-Timeframe Support**: Analysis across 5min, 10min, 15min, and 30min intervals
- **Volume Metrics**: Trading volume analysis and percentile ranking
- **Stability Assessment**: Volatility consistency measurement using coefficient of variation
- **Comprehensive Reporting**: Detailed analysis report with recommendations
- **Error Handling**: Robust error handling for API failures and missing data
- **Type Hints**: Full type annotation for better code quality
- **Documentation**: Comprehensive README with examples and configuration guides

### 🔧 Technical Improvements
- **CCXT Integration**: Enhanced exchange integration with rate limiting
- **Data Validation**: Automatic data cleaning and validation
- **Pandas Warnings**: Suppressed deprecated warnings for cleaner output
- **Modular Design**: Better code organization with separate functions
- **Configuration**: Flexible configuration through environment variables and constants

### 📊 Analysis Enhancements
- **Net Volatility**: Volatility calculation after spread deduction
- **Volume-Weighted Metrics**: Volume-adjusted volatility calculations
- **Composite Scoring**: Intelligent ranking system combining multiple factors
- **Risk Indicators**: Automatic detection of high spread or low stability warnings

### 🛠️ Developer Experience
- **Example Code**: Comprehensive example.py with usage demonstrations
- **Setup Script**: pip-installable package with setup.py
- **Git Integration**: Proper .gitignore and repository structure
- **License**: MIT license for open source distribution

### 📈 Performance
- **Efficient Data Fetching**: Optimized API calls with rate limiting
- **Memory Management**: Improved memory usage for large datasets
- **Error Recovery**: Graceful handling of network issues and missing data

### 🎯 User Experience
- **Rich Output**: Emoji-enhanced output with clear formatting
- **Progress Indicators**: Real-time progress updates during analysis
- **Clear Recommendations**: Actionable insights and warnings
- **Flexible Configuration**: Easy customization of symbols, exchanges, and weights

## [1.2.0] - 2025-08-26

### Fixed
- Replaced deprecated pandas resample alias "10T" with "10min"
- Silenced FutureWarning for pandas 3.0 compatibility

## [1.1.0] - Previous Version

### Added
- Basic volatility analysis using 10-minute windows
- CCXT integration for market data
- Simple ranking by volatility

### Features
- Single timeframe analysis (10-minute)
- Basic OHLCV data fetching
- Simple volatility calculation
- Exchange support via CCXT 